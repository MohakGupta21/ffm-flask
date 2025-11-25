from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union

app = FastAPI()

# --- CONFIGURATION ---
# The COLUMN_X_TOLERANCE is still present but ignored by get_group_references_v2
CONFIG = {
    'COLUMN_X_TOLERANCE': 0.10,       # Kept for completeness, but not used in V2 grouping
    'GROUP_SPACING_THRESHOLD': 0.08,  # 8% vertical gap breaks a group
    'EPSILON': 0.02,                  # 2% tolerance for perfect alignment detection
}

class LayoutRequest(BaseModel):
    reference_data: List[Dict[str, Any]]

# --- REVISED GROUPING LOGIC (Vertical Proximity Only) ---

def get_group_references_v2(flat_data, config):
    """
    Groups items based ONLY on vertical proximity (Top-to-Bottom flow).
    This function ignores horizontal position/columns.
    """
    if not flat_data:
        return []

    # 1. Sort the entire dataset by Top coordinate (Y-axis)
    vertical_sorted_data = sorted(flat_data, key=lambda x: x['BoundingBox']['Top'])
    
    final_groups = []
    current_group = []

    for item in vertical_sorted_data:
        # Extract BoundingBox for simpler access
        bbox = item['BoundingBox']
        top = bbox['Top']
        
        if not current_group:
            # Start of the first group
            current_group.append(item)
        else:
            last_item = current_group[-1]
            last_bbox = last_item['BoundingBox']
            last_item_bottom = last_bbox['Top'] + last_bbox['Height']
            
            # Calculate vertical gap
            vertical_gap = top - last_item_bottom
            
            if vertical_gap >= config['GROUP_SPACING_THRESHOLD']:
                # Gap is too big (start of a new block/paragraph)
                final_groups.append(current_group)
                current_group = [item]
            else:
                # Gap is small (continue the current block/paragraph)
                current_group.append(item)
                
    if current_group:
        final_groups.append(current_group)

    return final_groups

# --- ALIGNMENT LOGIC (Determines Alignment for the Group/Block) ---

def determine_alignment(group_items, config):
    """
    Determines the common alignment for a given group.
    """
    epsilon = config['EPSILON']
    
    # Logic for Multi-Item Groups
    if len(group_items) > 1:
        lefts = [item['BoundingBox']['Left'] for item in group_items]
        rights = [item['BoundingBox']['Left'] + item['BoundingBox']['Width'] for item in group_items]
        
        # Check variance (range) of edges
        left_variance = max(lefts) - min(lefts)
        right_variance = max(rights) - min(rights)
        
        if left_variance < epsilon:
            return "left_aligned"
        elif right_variance < epsilon:
            return "right_aligned"
        else:
            # If neither edge is aligned consistently, assume center alignment for the block
            return "center_aligned"
    
    # Logic for Single Item Groups (Heuristic based on canvas position)
    else:
        item = group_items[0]
        
        if item['BoundingBox']['Left'] < 0.30:
            return "left_aligned"
        elif item['BoundingBox']['Left'] > 0.60: 
            return "right_aligned"
        else:
            return "center_aligned"

# --- MAIN ENDPOINT ---

@app.post("/process-layout")
async def process_layout(payload: LayoutRequest):
    data = payload.dict()
    # Assuming we process the first item in the reference_data array
    ref_data = data['reference_data'][0]
    
    current_config = CONFIG.copy()
    current_config['IMAGE_W'] = data.get('image_w', 600)

    # --- 1. Flattening Phase (Including RTBs) ---
    flat_items = []

    # Helper function to safely add an item
    def add_flat_item(item, type_name):
        # Check if the item exists and has the required geometry
        if isinstance(item, dict) and 'BoundingBox' in item:
            item['_internal_type'] = type_name 
            flat_items.append(item)

    # 1.1 Extract Logo
    add_flat_item(ref_data.get('logo'), 'logo')

    # 1.2 Extract CTA
    add_flat_item(ref_data.get('CTA'), 'CTA')

    # 1.3 Extract Copies
    if ref_data.get('copies'):
        for copy_item in ref_data['copies']:
            add_flat_item(copy_item, 'copy')

    # 1.4 Extract RTBs (Handling both List and Dictionary structures)
    if ref_data.get('RTBs'):
        rtbs = ref_data['RTBs']
        
        if isinstance(rtbs, list):
            for item in rtbs:
                add_flat_item(item, 'RTB')
                    
        elif isinstance(rtbs, dict):
            for key, item in rtbs.items():
                # Item might be another nested dict, ensure it has BoundingBox
                if isinstance(item, dict): 
                    add_flat_item(item, 'RTB')

    # --- 2. Grouping Logic (Vertical V2) ---
    groups = get_group_references_v2(flat_items, current_config)

    # --- 3. Enrichment Logic ---
    group_counter = 2 # Start at g2
    
    for group_items in groups:
        group_id = f"g{group_counter}"
        
        # Determine the overall alignment for this vertical block
        alignment = determine_alignment(group_items, current_config)

        # Apply the group ID and alignment to all items in this block
        for item in group_items:
            item['group'] = group_id
            item['alignment'] = alignment
            
            # Clean up temporary internal keys
            if '_internal_type' in item: del item['_internal_type']

        group_counter += 2 # Increment (g2, g4, g6...)

    # --- 4. Return ---
    # Since we modified the dictionaries in place (by reference), 
    # 'ref_data' is already updated with 'group' and 'alignment'.
    return [ref_data]