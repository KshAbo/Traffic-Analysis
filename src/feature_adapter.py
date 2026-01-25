# src/feature_adapter.py

import pandas as pd
import os
from config import TRAIN_PATH, TEST_PATH, ID_COL

# 1. SETUP PATHS
# Use relative paths to be safe. Assumes files are in the 'src' folder or 'data' folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAMA_TRAIN_PATH = os.path.join(BASE_DIR, "train_features.csv")
TEAMA_TEST_PATH  = os.path.join(BASE_DIR, "test_features.csv")

# Map Teammate's camera names to Zindi's camera names
CAMERA_MAP = {
    "camera_1": "Norman Niles #1",
    "camera_2": "Norman Niles #2",
    "camera_3": "Norman Niles #3",
    "camera_4": "Norman Niles #4"
}

def clean_video_path(path):
    """
    Converts 'normanniles1/normanniles1_2025...mp4' 
    to just 'normanniles1_2025...mp4'
    """
    if pd.isna(path):
        return ""
    return str(path).split('/')[-1]

def adapt_features():
    print("--- 1. Loading Files ---")
    
    # Check if teammate's files exist
    if not os.path.exists(TEAMA_TRAIN_PATH):
        print(f"ERROR: Could not find {TEAMA_TRAIN_PATH}")
        print("Please move 'train_features.csv' into the 'src' folder.")
        return

    # Load Teammate's Data
    feat_train = pd.read_csv(TEAMA_TRAIN_PATH)
    feat_test  = pd.read_csv(TEAMA_TEST_PATH)
    print(f"Loaded {len(feat_train)} feature rows.")

    # Load Zindi Data (To get the ID Mapping)
    try:
        # We need the 'videos' column from Train.csv
        zindi_train = pd.read_csv(TRAIN_PATH)
        # Note: TestInputSegments often DOES NOT have the video filename.
        # If TestInputSegments.csv lacks 'videos', we might need 'test_metadata.csv'
        # But let's check if it exists.
        zindi_test = pd.read_csv(TEST_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find Zindi files at {TRAIN_PATH}")
        print("Make sure 'data/Train.csv' exists.")
        return

    print("--- 2. Cleaning Video Paths ---")
    
    # The column in Train.csv is called 'videos'
    if 'videos' in zindi_train.columns:
        zindi_train['clean_filename'] = zindi_train['videos'].apply(clean_video_path)
    else:
        print("CRITICAL: 'videos' column missing from Train.csv!")
        return

    # Check Test file for 'videos' column
    if 'videos' in zindi_test.columns:
        zindi_test['clean_filename'] = zindi_test['videos'].apply(clean_video_path)
    else:
        # Fallback: Try to merge Test on Camera + Time if filename is missing
        print("Warning: 'videos' column missing from Test CSV. Test merge might be tricky.")
        # For now, let's proceed with Train.
    
    print("--- 3. Merging (The Critical Step) ---")
    
    # MERGE TRAIN
    # We link Teammate's 'video_file' to Zindi's 'clean_filename'
    # We want to attach 'time_segment_id' (ID_COL) to the features.
    
    train_subset = zindi_train[[ID_COL, 'clean_filename']].drop_duplicates()
    
    train_ready = feat_train.merge(
        train_subset, 
        left_on='video_file', 
        right_on='clean_filename', 
        how='inner'
    )
    
    # MERGE TEST
    # (Assuming Test CSV has 'videos' column or similar structure)
    if 'clean_filename' in zindi_test.columns:
        test_subset = zindi_test[[ID_COL, 'clean_filename']].drop_duplicates()
        test_ready = feat_test.merge(
            test_subset, 
            left_on='video_file', 
            right_on='clean_filename', 
            how='inner'
        )
    else:
        print("Skipping Test Merge (Missing filename). You can only train, not submit.")
        test_ready = feat_test
    
    print(f"Success! Matched {len(train_ready)} training rows.")

    # --- 4. Format for train.py ---
    # We need to map the Camera IDs too
    train_ready['Camera'] = train_ready['camera_id'].map(CAMERA_MAP)
    if 'camera_id' in test_ready.columns:
        test_ready['Camera']  = test_ready['camera_id'].map(CAMERA_MAP)
    
    # Save to data folder
    out_train = "../data/cv_features_train_ready.csv"
    out_test  = "../data/cv_features_test_ready.csv"
    
    train_ready.to_csv(out_train, index=False)
    test_ready.to_csv(out_test, index=False)
    
    print(f"Saved fixed files to: {out_train}")
    print("DONE. You can now run train.py!")

if __name__ == "__main__":
    adapt_features()