from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI()

# --- CONFIGURATION ---
# These values control the sensitivity of the grouping and alignment logic.
CONFIG = {
    'COLUMN_X_TOLERANCE': 0.10,       # 10% of screen width difference allowed to be in same column
    'GROUP_SPACING_THRESHOLD': 0.08,  # 8% vertical gap breaks a group
    'EPSILON': 0.02,                  # 2% tolerance for perfect alignment detection
    # 'IMAGE_W': 600                    # Default width if not provided
}

class LayoutRequest(BaseModel):
    # input_file: str
    reference_data: List[Dict[str, Any]]

# --- HELPER LOGIC ---

def get_group_references(flat_data, config):
    """
    Groups items based on column alignment first, then vertical proximity.
    """
    # 1. Sort by Left to find columns
    left_sorted_data = sorted(flat_data, key=lambda x: x['BoundingBox']['Left'])
    
    columns = []
    for item in left_sorted_data:
        left = item['BoundingBox']['Left']
        assigned = False
        
        # Check against existing columns
        for column in columns:
            # We compare against the average Left of the column or just the first item
            first_left = column[0]['BoundingBox']['Left']
            if abs(left - first_left) < config['COLUMN_X_TOLERANCE']:
                column.append(item)
                assigned = True
                break
        
        if not assigned:
            columns.append([item])

    final_groups = []

    # 2. Process each column for vertical grouping
    for column in columns:
        # Sort items in this column from Top to Bottom
        column_sorted_by_top = sorted(column, key=lambda x: x['BoundingBox']['Top'])
        
        current_group = []
        for item in column_sorted_by_top:
            top = item['BoundingBox']['Top']
            
            if not current_group:
                current_group.append(item)
            else:
                last_item = current_group[-1]
                last_item_bottom = last_item['BoundingBox']['Top'] + last_item['BoundingBox']['Height']
                
                # Calculate vertical gap
                vertical_gap = top - last_item_bottom
                
                if vertical_gap >= config['GROUP_SPACING_THRESHOLD']:
                    # Gap is too big, close current group and start new one
                    final_groups.append(current_group)
                    current_group = [item]
                else:
                    # Gap is small, add to current group
                    current_group.append(item)
                    
        if current_group:
            final_groups.append(current_group)

    # 3. Sort final groups by their top position (Visual flow)
    final_groups.sort(key=lambda g: g[0]['BoundingBox']['Top'])
    return final_groups

def determine_alignment(group_items, config):
    """
    Determines if a group is Left, Right, or Center aligned.
    """
    # Logic for Single Item Groups
    if len(group_items) == 1:
        item = group_items[0]
        center_x = item['BoundingBox']['Left'] + (item['BoundingBox']['Width'] / 2)
        
        # If purely centered (within tolerance)
        if abs(center_x - 0.5) < config['EPSILON']:
            return "center_aligned"
        
        # Otherwise, guess based on canvas position
        if item['BoundingBox']['Left'] < 0.30:
            return "left_aligned"
        elif item['BoundingBox']['Left'] > 0.60: # Adjusted for the specific example where logo is far right
            return "right_aligned"
        else:
            return "center_aligned"

    # Logic for Multi-Item Groups
    lefts = [item['BoundingBox']['Left'] for item in group_items]
    rights = [item['BoundingBox']['Left'] + item['BoundingBox']['Width'] for item in group_items]
    
    left_variance = max(lefts) - min(lefts)
    right_variance = max(rights) - min(rights)
    
    if left_variance < config['EPSILON']:
        return "left_aligned"
    elif right_variance < config['EPSILON']:
        return "right_aligned"
    else:
        return "center_aligned"

# --- MAIN ENDPOINT ---

@app.post("/process-layout")
async def process_layout(payload: LayoutRequest):
    data = payload.dict()
    ref_data = data['reference_data'][0] # Assuming processing first item in array
    
    # Update Config from Input if available
    current_config = CONFIG.copy()
    # current_config['IMAGE_W'] = data.get('image_w', 600)

    # 1. Flatten the Data for Processing
    # We need a unified list of objects to run the clustering logic on
    flat_items = []

    # Extract Logo
    if ref_data.get('logo'):
        # Store reference to the original object to modify it later
        item = ref_data['logo']
        item['_internal_type'] = 'logo' 
        flat_items.append(item)
    
    if ref_data.get('RTBs'):
        # Store reference to the original object to modify it later
        item = ref_data['RTBs']
        item['_internal_type'] = 'RTBs' 
        flat_items.append(item)

    # Extract CTA
    if ref_data.get('CTA'):
        item = ref_data['CTA']
        item['_internal_type'] = 'CTA'
        flat_items.append(item)

    # Extract Copies
    if ref_data.get('copies'):
        for idx, copy_item in enumerate(ref_data['copies']):
            copy_item['_internal_type'] = 'copy'
            copy_item['_internal_idx'] = idx
            flat_items.append(copy_item)

    # 2. Run Grouping Logic
    groups = get_group_references(flat_items, current_config)

    # 3. Assign Groups and Alignment
    group_counter = 2 # Starting at g2 to match your example style, or use 1
    
    for group_items in groups:
        group_id = f"g{group_counter}"
        
        # Detect Alignment for this specific group
        alignment = determine_alignment(group_items, current_config)

        # Apply to all items in this group
        for item in group_items:
            item['group'] = group_id
            item['alignment'] = alignment
            
            # Remove temporary internal keys before output
            if '_internal_type' in item: del item['_internal_type']
            if '_internal_idx' in item: del item['_internal_idx']

        group_counter += 2 # Increment (g2, g4, g6...)

    # 4. Format Output
    # Since we modified the dictionaries in place (by reference), 
    # 'ref_data' is already updated.
    
    output = [ref_data]
    
    return output

# To run locally:
# uvicorn main:app --reload