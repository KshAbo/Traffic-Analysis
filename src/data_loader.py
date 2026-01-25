import pandas as pd
import re
from config import TRAIN_PATH, TEST_PATH, ID_COL

def extract_camera_from_id(row_id):
    """
    Extracts 'Norman Niles #1' from 'time_segment_0_Norman Niles #1_congestion_enter_rating'
    """
    # Look for the pattern "Norman Niles #X"
    match = re.search(r'(Norman Niles #\d+)', str(row_id))
    if match:
        return match.group(1)
    return "Unknown"

def load_and_aggregate():
    print("--- Loading Raw Data ---")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    # 1. FIX TRAIN: Rename 'view_label' to 'Camera'
    if 'view_label' in train.columns:
        train = train.rename(columns={'view_label': 'Camera'})
    else:
        # Fallback if view_label is missing (unlikely in Zindi data)
        print("Warning: 'view_label' not found in Train.csv. Attempting extraction from ID...")
        train['Camera'] = train[ID_COL].apply(extract_camera_from_id)

    # 2. FIX TEST: Extract 'Camera' from the ID string
    # TestInputSegments.csv usually does NOT have 'view_label', only 'ID'
    if 'view_label' not in test.columns:
        # We must extract the camera name from the ID column to merge features
        # ID Format: time_segment_129_Norman Niles #1_congestion_enter_rating
        # Note: If ID_COL is 'time_segment_id', we might need the full string ID.
        # Let's check if there is a column with the full ID string (usually 'ID' or similar)
        
        # If the test csv only has 'time_segment_id', we have a problem.
        # But Zindi usually gives 'ID' column.
        col_to_parse = 'ID' if 'ID' in test.columns else ID_COL
        
        test['Camera'] = test[col_to_parse].apply(extract_camera_from_id)
    else:
        test = test.rename(columns={'view_label': 'Camera'})

    # 3. Sort for Time Series safety
    if 'time_segment_id' in train.columns:
        train = train.sort_values(by=['Camera', 'time_segment_id']).reset_index(drop=True)
    if 'time_segment_id' in test.columns:
        test = test.sort_values(by=['Camera', 'time_segment_id']).reset_index(drop=True)

    print(f"Loaded Train: {train.shape}, Test: {test.shape}")
    print(f"Columns in Train: {train.columns.tolist()}")
    return train, test