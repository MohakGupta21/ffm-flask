import json
from flask import Flask, request, jsonify
import subprocess
import copy
import requests 
import os       
try:
    # Requires 'pip install Pillow'
    from PIL import ImageFont
except ImportError:
    print("WARNING: Pillow (PIL) is not installed. Font fitting logic will fail.")
    # Define a mock class/function to prevent crash if running without Pillow
    class MockImageFont:
        @staticmethod
        def truetype(font_path, size):
            return MockFont()
    class MockFont:
        def getbbox(self, text):
            # Return a mock width of 50px per char for fallback
            return (0, 0, len(text) * 50, 0)
    ImageFont = MockImageFont
    

app = Flask(__name__)

# --- Configuration (CONSTANTS: These define default/fallback values and fixed thresholds) ---
CONFIG = {
    # Dynamic values now use placeholders or fallbacks
    "IMAGE_W": 1024, # FALLBACK: Overridden by client input
    "IMAGE_H": 1024, # FALLBACK: Overridden by client input
    "INPUT_FILE": "file.png", # FIXED LOCAL PATH for the downloaded S3 image
    
    "LINE_SPACING_PIXELS": 4, 
    "OUTPUT_FILE": "output_with_text_new.png", # FIXED PATH on the server
    "FONT_COLOR": "black",
    "EPSILON": 0.005, # Fixed threshold
    "GROUP_SPACING_THRESHOLD": 0.02, # Fixed threshold
    "COLUMN_X_TOLERANCE": 0.20, # Fixed threshold
    
    # --- CRITICAL: UPDATE THIS PATH ---
    # Change this placeholder to the absolute path where your TTF files are stored.
    "FONT_BASE_PATH": "/Users/vanisha/Desktop/static-ads/python-scripts/fonts" 
}

# ----------------------------------------------------
# --- NEW: FONT SIZING LOGIC ---
# ----------------------------------------------------

def get_fitted_font_metrics(text_data, image_w, font_base_path):
    """
    Calculates the final font size, ensuring the text only shrinks if it
    goes outside the canvas boundaries, based on its specific alignment type.

    Args:
        text_data (dict): Contains 'text', 'font_path' (full filename), 'font_size', 
                          'reference_data', and 'placement_type'.
        image_w (int): The width of the canvas/image in pixels.
        font_base_path (str): The directory containing font files.

    Returns:
        dict: Updated metrics including 'final_font_size'.
    """
    
    text = text_data['text']
    font_file_name = text_data['font_path']
    initial_font_size = text_data['font_size']
    placement_type = text_data['placement_type']
    
    ref_bbox = text_data['reference_data']['Geometry']['BoundingBox']
    
    # Normalized anchor points (0.0 to 1.0)
    ref_left_min_norm = ref_bbox['Left'] 
    ref_right_max_norm = ref_bbox['Left'] + ref_bbox['Width']
    
    # Pixel anchor points (used for canvas boundary checks)
    ref_left_min_px = int(ref_left_min_norm * image_w)
    ref_right_max_px = int(ref_right_max_norm * image_w)
    
    # Centroid X (used for CENTER ALIGNED check)
    centroid_x_norm = ref_bbox['Left'] + (ref_bbox['Width'] / 2)
    centroid_x_px = int(centroid_x_norm * image_w)
    
    # 2. Resolve Full Font Path
    font_file = os.path.join(font_base_path, font_file_name)

    # 3. Iteratively adjust font size to fit 
    current_size = initial_font_size
    max_iterations = 30 # Safety break
    
    while max_iterations > 0:
        try:
            # Load font at the current size
            font = ImageFont.truetype(font_file, int(current_size))
            
            # Measure the text width (right - left of bounding box)
            bbox = font.getbbox(text)
            text_width_px = bbox[2] - bbox[0]

        except Exception as e:
            # Fallback if font loading/measurement fails 
            print(f"Error loading font or measuring: {e}. Falling back to initial size.")
            text_width_px = 0
            break

        # --- FITTING CRITERIA (Break if text fits canvas boundary) ---
        if placement_type == 'CENTER_ALIGNED_CANVAS':
            # Text is centered on centroid, check both canvas edges
            potential_left_bound = centroid_x_px - (text_width_px / 2)
            potential_right_bound = centroid_x_px + (text_width_px / 2)

            if potential_left_bound >= 0 and potential_right_bound <= image_w:
                break
        
        elif placement_type == 'LEFT_ALIGNED_GROUP':
            # Text is anchored to the left (ref_left_min_px), check right canvas edge
            potential_right_bound = ref_left_min_px + text_width_px
            if potential_right_bound <= image_w:
                break
            
        elif placement_type == 'RIGHT_ALIGNED_GROUP':
            # Text is anchored to the right (ref_right_max_px), check left canvas edge
            potential_left_bound = ref_right_max_px - text_width_px
            if potential_left_bound >= 0:
                break
        
        else: # Fallback for UNKNOWN/other types - use initial size if it fits roughly
              # This uses the previous width-based check as a final safety measure
              ref_width_px = ref_bbox['Width'] * image_w
              if text_width_px <= ref_width_px * 1.05:
                break
            
        # If the break condition wasn't met (i.e., it went past a boundary), decrease size.
        current_size *= 0.95 

        max_iterations -= 1

    # Final font size, rounded for clean FFmpeg output
    final_font_size = round(max(current_size, 1), 2) # Ensure size is at least 1
    
    return {
        "final_font_size": final_font_size
    }

# ----------------------------------------------------
# --- HELPER AND CORE LOGIC FUNCTIONS ---
# ----------------------------------------------------

def download_file(url, local_filename):
    """Downloads a file from a URL (e.g., S3 URL) to a local path."""
    print(f"Attempting to download input image from {url} to {local_filename}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status() 
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print("Download successful.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed for {url}: {e}")
        return False

def get_group_references(ref_data_with_indices, config): 
    valid_data = [item for item in ref_data_with_indices if 'Geometry' in item and 'BoundingBox' in item['Geometry']]
    if not valid_data: return []

    left_sorted_data = sorted(valid_data, key=lambda x: x['Geometry']['BoundingBox']['Left'])
    
    columns = [] 
    
    for item in left_sorted_data:
        left = item['Geometry']['BoundingBox']['Left']
        assigned = False
        
        for column in columns:
            first_left = column[0]['Geometry']['BoundingBox']['Left']
            if abs(left - first_left) < config['COLUMN_X_TOLERANCE']: 
                column.append(item)
                assigned = True
                break
        
        if not assigned:
            columns.append([item])
            
    final_groups = []

    for column in columns:
        column_sorted_by_top = sorted(column, key=lambda x: x['Geometry']['BoundingBox']['Top'])
        
        current_group = []
        for item in column_sorted_by_top:
            top = item['Geometry']['BoundingBox']['Top']
            
            if not current_group:
                current_group.append(item)
            else:
                last_item = current_group[-1]
                last_item_bottom = last_item['Geometry']['BoundingBox']['Top'] + last_item['Geometry']['BoundingBox']['Height']
                vertical_proximity = (top - last_item_bottom)
                
                if vertical_proximity >= config['GROUP_SPACING_THRESHOLD']: 
                    final_groups.append(current_group)
                    current_group = [item]
                else:
                    current_group.append(item)
                    
        if current_group:
            final_groups.append(current_group)

    final_groups.sort(key=lambda g: g[0]['Geometry']['BoundingBox']['Top'])
    
    return final_groups

def detect_alignment(ref_group, tolerance): 
    if not ref_group: return 'UNKNOWN'

    lefts = [item['Geometry']['BoundingBox']['Left'] for item in ref_group]
    rights = [item['Geometry']['BoundingBox']['Left'] + item['Geometry']['BoundingBox']['Width'] for item in ref_group]
    
    left_range = max(lefts) - min(lefts)
    right_range = max(rights) - min(rights)
    
    if left_range < tolerance:
        return 'LEFT_ALIGNED_GROUP'
    if right_range < tolerance:
        return 'RIGHT_ALIGNED_GROUP'
    
    return 'CENTER_ALIGNED_CANVAS'

def get_group_analysis(ref_data_with_indices, new_texts, config):
    all_ref_groups = get_group_references(ref_data_with_indices, config)
    analyses = []

    for i, ref_group in enumerate(all_ref_groups):
        
        ref_count = len(ref_group)
        
        ref_top_min = min([item['Geometry']['BoundingBox']['Top'] for item in ref_group])
        ref_left_min = min([item['Geometry']['BoundingBox']['Left'] for item in ref_group])
        ref_right_max = max([item['Geometry']['BoundingBox']['Left'] + item['Geometry']['BoundingBox']['Width'] for item in ref_group])
        
        ref_centroid_x_norm = ref_left_min + ((ref_right_max - ref_left_min) / 2)
        group_center_x_anchor_pixel = int(ref_centroid_x_norm * config['IMAGE_W'])

        if ref_count == 1:
            ref_item = ref_group[0]
            bbox = ref_item['Geometry']['BoundingBox']
            
            if abs(ref_centroid_x_norm - 0.5) < config['EPSILON']: 
                placement = 'CENTER_ALIGNED_CANVAS'
            else:
                x_left_norm = bbox['Left']
                
                if x_left_norm < 0.3:
                    placement = 'LEFT_ALIGNED_GROUP'
                elif x_left_norm >= 0.70:
                    placement = 'RIGHT_ALIGNED_GROUP'
                else: 
                    placement = 'CENTER_ALIGNED_CANVAS'
        else:
            placement = detect_alignment(ref_group, config['EPSILON'])

        max_ref_width_norm = 0
        max_ref_chars = 0
        
        for item in ref_group:
            current_chars = len(item['DetectedText'])
            if current_chars > max_ref_chars:
                 max_ref_chars = current_chars
                 max_ref_width_norm = item['Geometry']['BoundingBox']['Width'] 
            
        ref_char_width_norm = max_ref_width_norm / max_ref_chars if max_ref_chars > 0 else 0
        
        group_ref_indices = [item['original_index'] for item in ref_group]
        max_new_chars = 0
        for original_idx in group_ref_indices:
            if original_idx < len(new_texts):
                new_text_item = new_texts[original_idx]
                max_new_chars = max(max_new_chars, len(new_text_item['text']))

        new_group_width_norm = ref_char_width_norm * max_new_chars
        
        fixed_x_left_anchor_norm = ref_left_min
        fixed_x_left_anchor_pixel = int(fixed_x_left_anchor_norm * config['IMAGE_W'])
        
        fixed_x_right_anchor_norm = ref_right_max
      
        fixed_x_right_anchor_pixel = int(fixed_x_right_anchor_norm * config['IMAGE_W'])

        analyses.append({
            "placement_type": placement,
            "ref_top_min": ref_top_min,
            "ref_left_min": ref_left_min, 
            "ref_right_max": ref_right_max,
            "group_id": f"GROUP_{i}",
            "ref_count": ref_count,
            "center_x_anchor": group_center_x_anchor_pixel,
            "fixed_x_left_anchor": fixed_x_left_anchor_pixel,
            "fixed_x_right_anchor": fixed_x_right_anchor_pixel,
            "original_indices": group_ref_indices 
        })

    return analyses

def assign_new_texts_to_groups(new_texts, analyses):
    index_to_analysis = {}
    for analysis in analyses:
        for original_idx in analysis['original_indices']:
            index_to_analysis[original_idx] = analysis
            
    pre_sorted_texts = []
    for i, new_text_item in enumerate(new_texts):
        if i in index_to_analysis:
            analysis = index_to_analysis[i]
            
            line = new_text_item.copy()
            line['group_id'] = analysis['group_id']
            line['analysis'] = analysis 
            line['original_index'] = i 
            
            stack_order = analysis['original_indices'].index(i) 
            line['stack_order'] = stack_order
            
            pre_sorted_texts.append(line)
            
    assigned_texts = sorted(pre_sorted_texts, key=lambda x: (x['group_id'], x['stack_order']))
    
    return assigned_texts

def escape_ffmpeg_text(text):
    """Robustly escapes text for FFmpeg drawtext."""
    text = text.replace(':', '\\:')
    text = text.replace(',', '\\,')
    text = text.replace("'", "\\'") 
    return text

def generate_ffmpeg_command(assigned_texts, config):
    """
    Generates the cumulative FFmpeg command using the simple -vf filter chain.
    Uses the pre-calculated final_font_size from the assigned_texts list.
    """
    filters = []
    y_positions = {}
    
    for line in assigned_texts:
        analysis = line['analysis']
        group_id = line['group_id']
        escaped_text = escape_ffmpeg_text(line['text'])
        
        # --- 1. Calculate Y Position (Stacking) ---
        if group_id in y_positions:
            prev_line_idx = -1
            for idx, item in enumerate(assigned_texts):
                if item['group_id'] == group_id and item['stack_order'] == line['stack_order'] - 1:
                    prev_line_idx = idx
                    break
            
            # Use the previous line's *final, fitted* font size for spacing
            prev_font_size = assigned_texts[prev_line_idx].get('final_font_size', line['font_size']) if prev_line_idx != -1 else line['font_size']
            
            current_abs_y = y_positions[group_id] + prev_font_size + config['LINE_SPACING_PIXELS']
        else:
            current_abs_y = int(analysis['ref_top_min'] * config['IMAGE_H'])
        
        y_positions[group_id] = current_abs_y

        # --- 2. Calculate X Position (Alignment Logic) ---
        placement_type = analysis['placement_type']
        
        if placement_type == 'LEFT_ALIGNED_GROUP':
            # Use ref_left_min as the anchor point
            abs_x = analysis['fixed_x_left_anchor']
        elif placement_type == 'RIGHT_ALIGNED_GROUP':
            # Use ref_right_max as the anchor point
            abs_x = f"{analysis['fixed_x_right_anchor']}-text_w"
        elif placement_type == 'CENTER_ALIGNED_CANVAS':
            # Use center_x_anchor (based on reference bounding box center)
            abs_x = f"{analysis['center_x_anchor']}-(text_w/2)" 
        else:
            # Fallback to left-aligned start if placement is UNKNOWN or single
            abs_x = int(analysis['ref_left_min'] * config['IMAGE_W'])
        
        # Use the newly calculated, fitted font size
        final_font_size = line.get('final_font_size', line['font_size'])
        print(f"final_font_size: {final_font_size}")
        print(f"line['font_size']: {line['font_size']}")

        # Construct the full font path using the full filename from input
        font_file_name = line['font_path']
        full_font_path = os.path.join(config['FONT_BASE_PATH'], font_file_name)
        
        # --- 3. Filter Construction ---
        filter_str = (
            f"drawtext=fontfile='{full_font_path}':"
            f"text='{escaped_text}':"
            f"fontcolor={line['color']}:"
            f"fontsize={final_font_size}:" # Using the fitted size
            f"x={abs_x}:"
            f"y={current_abs_y}"
        )
        filters.append(filter_str)

    vf_filter = ','.join(filters)
    
    command = [
        'ffmpeg',
        '-i', config['INPUT_FILE'], 
        '-vf', vf_filter,
        '-frames:v', '1',
        '-y',
        config['OUTPUT_FILE']
    ]
    return command


# ----------------------------------------------------
# --- Flask App Endpoints ---
# ----------------------------------------------------

def process_image_with_ffmpeg(input_data):
    """
    Receives dynamic data from the API and executes the full FFmpeg text replacement logic.
    """
    
    # 1. Extract dynamic configuration and core data
    image_w = input_data.get('image_w')
    image_h = input_data.get('image_h')
    s3_url = input_data.get('input_file') 
    reference_data = input_data.get('reference_data', [])
    # new_texts = input_data.get('new_texts', [])
    print(reference_data)
    # Validation
    if not all([image_w, image_h, s3_url]):
        missing = [k for k, v in {'image_w': image_w, 'image_h': image_h, 'input_file (S3 URL)': s3_url}.items() if not v]
        return {"status": "error", "message": f"Missing required dynamic configuration parameters: {', '.join(missing)}."}, 400

    if not reference_data or not new_texts:
        return {"status": "error", "message": "Missing 'reference_data' or 'new_texts' in payload."}, 400

    # 2. Create runtime configuration and download input file
    current_config = CONFIG.copy()
    current_config['IMAGE_W'] = int(image_w)
    current_config['IMAGE_H'] = int(image_h)

    # 2.1. Download the input image from the S3 URL to a local path for FFmpeg
    if not download_file(s3_url, current_config['INPUT_FILE']):
        return {"status": "error", "message": f"Failed to download input file from {s3_url}."}, 500
    
    # 3. Core Logic Execution
    
    # 3.1. Index the reference data 
    indexed_ref_data = []
    # Map the reference data for easy lookup by Tag (or by index)
    # ref_data_map = {item.get('Tag'): item for item in reference_data} 
    
    for idx, item in enumerate(ref_data_map):
        temp = copy.deepcopy(item) 
        temp['original_index'] = idx
        indexed_ref_data.append(temp)

    print(indexed_ref_data)
    # 3.2. Analyze the reference data and create group analysis
    group_analyses = get_group_analysis(indexed_ref_data, new_texts, current_config)
    if group_analyses:
        print(f"**Placement Analysis Results (Reverted Single-Line Alignment, EPSILON {CONFIG['EPSILON']}):**")
        for a in group_analyses:
            print(f"[{a['group_id']}] Alignment: **{a['placement_type']}**, Lines: {a['ref_count']}, Indices: {a['original_indices']}")
            print(f"    Centroid X (Pixel): {a['center_x_anchor']}")
            print(f"    Fixed LEFT X Anchor (Pixel): {a['fixed_x_left_anchor']}")
            print(f"    Fixed RIGHT X Anchor (Pixel): {a['fixed_x_right_anchor']}")
        print("-" * 50)

    if not group_analyses:
        return {"status": "error", "message": "Could not determine text alignment groups from reference data."}, 400
    
    # 3.3. Assign new texts to the dynamically created groups
    assigned_texts = assign_new_texts_to_groups(new_texts, group_analyses)
    
    # 3.4. Apply Font Fit Check and update font size for each line
    for line in assigned_texts:
        # We need the original reference item's BoundingBox for the fit check
        # Use the original index to look up the reference data
        original_index = line['original_index']
        ref_item = indexed_ref_data[original_index]
        
        if ref_item and line['text']:
            # Get the determined placement type for this individual line
            group_id = line['group_id']
            analysis = next(a for a in group_analyses if a['group_id'] == group_id)
            # print("analysis: ", analysis)
            placement_type = analysis['placement_type']
            # print("placement_type: ", placement_type)

            text_data_for_fit = {
                'text': line['text'],
                'font_path': line['font_path'],
                'font_size': line['font_size'],
                'reference_data': ref_item,
                'placement_type': placement_type # Pass placement type to logic
            }
            
            # This call shrinks the font size if the text is too wide for the canvas boundaries
            fitted_metrics = get_fitted_font_metrics(
                text_data_for_fit, 
                current_config['IMAGE_W'], 
                current_config['FONT_BASE_PATH']
            )
            # Store the new, fitted font size back into the line dictionary
            line['final_font_size'] = fitted_metrics['final_font_size']
        else:
            # Fallback to the initial size if data is missing or text is empty
            line['final_font_size'] = line['font_size']
            
    # 3.5. Generate the FFmpeg command
    ffmpeg_cmd_list = generate_ffmpeg_command(assigned_texts, current_config)

    # 4. Execute FFmpeg command
    try:
        print("\n--- Executing FFmpeg Command ---")
        print(" ".join(str(x) for x in ffmpeg_cmd_list))
        
        # NOTE: Uncomment the subprocess run in your local environment
        subprocess.run(ffmpeg_cmd_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # print(" successful FFmpeg execution.")
        
        return {
            "status": "success",
            "output_file_path": current_config['OUTPUT_FILE'],
            "details": f"Successfully generated FFmpeg command and (MOCK) executed it. Image saved locally.",
            "command": " ".join(str(x) for x in ffmpeg_cmd_list)
        }, 200

    except subprocess.CalledError as e:
        print(f"❌ FFmpeg failed with error code {e.returncode}. STDERR: {e.stderr.decode('utf-8')}")
        return {
            "status": "error",
            "message": f"FFmpeg execution failed.",
            "details": e.stderr.decode('utf-8')
        }, 500
    except FileNotFoundError:
        print("❌ FFmpeg executable not found.")
        return {
            "status": "error",
            "message": "FFmpeg executable not found. Make sure it's installed and in your PATH."
        }, 500
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return {
            "status": "error",
            "message": f"An unexpected error occurred during processing: {e}"
        }, 500


@app.route('/process_image', methods=['POST'])
def handle_image_process():
    """API endpoint to receive data from n8n and trigger processing."""
    
    try:
        input_data = request.json
        if not input_data:
            return jsonify({"status": "error", "message": "No JSON data received."}), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to parse JSON input: {e}"}), 400

    response_data, status_code = process_image_with_ffmpeg(input_data)
    
    return jsonify(response_data), status_code

# To run the app locally: python app.py
if __name__ == '__main__':
    print("Starting Flask API on http://127.0.0.1:5001/process_image")
    app.run(debug=True, host='0.0.0.0', port=5001)
