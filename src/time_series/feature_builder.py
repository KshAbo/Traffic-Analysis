# src/feature_builder.py

import pandas as pd

def add_rolling_features(df, feature_cols):
    """
    Adds Time-Series context:
    1. Rolling Averages (Trend)
    2. Lags (History)
    3. Deltas (Rate of Change)
    """
    print("--- Engineering Time-Series Features ---")
    
    # Sort specifically by Camera and Time to ensure order
    df = df.sort_values(by=['Camera', 'time_segment_id']).reset_index(drop=True)
    
    # We will engineer features for these specific columns
    # (Focus on the most important ones from your teammate's file)
    target_cols = ['mean_vehicle_count', 'avg_density', 'mean_displacement', 'stop_ratio']
    
    # Filter to only existing columns
    target_cols = [c for c in target_cols if c in df.columns]

    for col in target_cols:
        # 1. Rolling Stats (Window of 5 minutes)
        # Tells the model: "Is this a busy 5-minute block?"
        df[f'{col}_roll_mean_5'] = df.groupby('Camera')[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{col}_roll_std_5']  = df.groupby('Camera')[col].transform(lambda x: x.rolling(5, min_periods=1).std())
        
        # 2. Lag Features (What happened 1 minute ago?)
        # Tells the model: "Instant context"
        df[f'{col}_lag_1'] = df.groupby('Camera')[col].shift(1)
        df[f'{col}_lag_2'] = df.groupby('Camera')[col].shift(2)
        
        # 3. Delta (Rate of Change)
        # Tells the model: "Is traffic rapidly increasing?" (Current - Previous)
        df[f'{col}_delta_1'] = df[col] - df[f'{col}_lag_1']

    # Fill NaNs created by lags (first few rows) with 0 or mean
    df = df.fillna(0)
    
    return df

def merge_cv_features(base_df, cv_df):
    # (Keep your existing merge logic here)
    # Ensure you map 'Camera' and 'time_segment_id' correctly before merging
    if 'Camera' not in cv_df.columns and 'camera_id' in cv_df.columns:
        # Quick Fix map if missing
        cam_map = {"camera_1": "Norman Niles #1", "camera_2": "Norman Niles #2", 
                   "camera_3": "Norman Niles #3", "camera_4": "Norman Niles #4"}
        cv_df['Camera'] = cv_df['camera_id'].map(cam_map)
        
    merged = pd.merge(base_df, cv_df, on=['time_segment_id', 'Camera'], how='left')
    return merged