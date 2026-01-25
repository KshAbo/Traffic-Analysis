import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SHIFT = 7 
SEED = 42

# --- 1. NUCLEAR FEATURE ENGINEERING ---
def create_global_features(df):
    """
    Calculates the 'Pulse' of the entire roundabout.
    If Camera 1 is jammed, Camera 2 is likely affected.
    """
    # Group by time to get the average state of ALL cameras
    global_stats = df.groupby('time_segment_id').agg({
        'mean_vehicle_count': ['mean', 'max'],
        'jam_factor': ['mean', 'max']
    })
    global_stats.columns = ['_'.join(col).strip() + '_global' for col in global_stats.columns.values]
    df = df.merge(global_stats, on='time_segment_id', how='left')
    return df

def feature_engineering(df):
    # 1. Physics Features
    # Pressure: High positive pressure = Jam accumulation
    df['pressure'] = df['entry_count'] - df['exit_count']
    # Jam Factor: Density * Stop Ratio (The most predictive feature)
    df['jam_factor'] = df['mean_vehicle_count'] * df['stop_ratio']
    # Flow Efficiency: Speed / Density
    df['efficiency'] = df['mean_displacement'] / (df['mean_vehicle_count'] + 1.0)
    
    # 2. Time Features
    if 'video_time' in df.columns:
        dt = pd.to_datetime(df['video_time'], errors='coerce')
        hour = dt.dt.hour.fillna(12)
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    else:
        df['hour_sin'] = 0; df['hour_cos'] = 0

    # 3. Camera Encoding
    cam_map = {"Norman Niles #1": 1, "Norman Niles #2": 2, "Norman Niles #3": 3, "Norman Niles #4": 4}
    if 'Camera' in df.columns:
        df['camera_code'] = df['Camera'].map(cam_map).fillna(0)

    # 4. Rolling Trends (The "Memory")
    df = df.sort_values(by=['Camera', 'time_segment_id'])
    grp = df.groupby('Camera')
    
    # Calculate trends over the last 5 minutes
    for col in ['mean_vehicle_count', 'jam_factor', 'pressure']:
        df[f'{col}_roll_mean'] = grp[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
        # Acceleration: Is it getting worse FASTER?
        df[f'{col}_accel'] = df[col] - df[f'{col}_roll_mean']

    return df

# --- 2. THE ENSEMBLE (Optimized for Stability) ---
def get_stable_ensemble():
    # XGBoost: The main brain
    xgb_clf = xgb.XGBClassifier(
        n_estimators=700, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED,
        tree_method='hist'
    )
    
    # LightGBM: The speedster
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=700, num_leaves=31, learning_rate=0.03,
        class_weight='balanced', n_jobs=-1, random_state=SEED, verbose=-1
    )
    
    # Random Forest: The stabilizer (prevents overfitting)
    rf_clf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=4,
        class_weight='balanced_subsample', n_jobs=-1, random_state=SEED
    )
    
    # Soft Voting returns Probabilities, which allows us to threshold later
    return VotingClassifier(
        estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('rf', rf_clf)],
        voting='soft', weights=[1.5, 1.0, 1.0]
    )

def apply_thresholds(probs, label_map):
    """
    CUSTOM POST-PROCESSING:
    If the model is >30% sure it's 'Heavy Delay' (Class 3), we trust it.
    Default argmax would require >25% in a 4-class problem, but we want to be aggressive on jams.
    """
    # Classes: 0=Free, 1=Light, 2=Moderate, 3=Heavy
    final_preds = []
    
    # Get numeric index for Heavy Delay
    heavy_idx = label_map["heavy delay"]
    moderate_idx = label_map["moderate delay"]
    
    for row_probs in probs:
        # Aggressive thresholding for rare classes
        if row_probs[heavy_idx] > 0.28: # Lower threshold to catch more jams
            final_preds.append(heavy_idx)
        elif row_probs[moderate_idx] > 0.35:
            final_preds.append(moderate_idx)
        else:
            # Otherwise, take the max as usual
            final_preds.append(np.argmax(row_probs))
            
    return final_preds

# --- 3. MAIN PIPELINE ---
def main():
    print("--- 1. LOADING ---")
    try:
        path = "../data/"
        train_feats = pd.read_csv(path + 'cv_features_train_ready.csv')
        test_feats = pd.read_csv(path + 'cv_features_test_ready.csv')
        train_labels = pd.read_csv(path + 'Train.csv')
        test_meta = pd.read_csv(path + 'TestInputSegments.csv')
        sample_sub = pd.read_csv(path + 'SampleSubmission.csv')
    except:
        train_feats = pd.read_csv('cv_features_train_ready.csv')
        test_feats = pd.read_csv('cv_features_test_ready.csv')
        train_labels = pd.read_csv('Train.csv')
        test_meta = pd.read_csv('TestInputSegments.csv')
        sample_sub = pd.read_csv('SampleSubmission.csv')

    train_labels.rename(columns={'view_label': 'Camera'}, inplace=True)
    if 'view_label' in test_meta.columns: test_meta.rename(columns={'view_label': 'Camera'}, inplace=True)

    train_df = pd.merge(train_feats, train_labels, on=['time_segment_id', 'Camera'], how='inner')
    test_df = pd.merge(test_feats, test_meta, on=['time_segment_id', 'Camera'], how='left')

    print("--- 2. ENGINEERING ---")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    train_df = create_global_features(train_df)
    test_df = create_global_features(test_df)

    # Clean Cols
    exclude = ['camera_id', 'video_file', 'clean_filename', 'Camera', 
               'congestion_enter_rating', 'congestion_exit_rating',
               'target_future', 'time_segment_id', 'minute', 'ID',
               'dt', 'video_time', 'videos', 'date', 'datetimestamp_start', 'datetimestamp_end',
               'responseId', 'ID_enter', 'ID_exit', 'signaling', 'cycle_phase', 'target_temp']
    feature_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
    print(f"Features: {len(feature_cols)}")

    # Setup Targets
    label_map = {"free flowing": 0, "light delay": 1, "moderate delay": 2, "heavy delay": 3}
    reverse_map = {v: k for k, v in label_map.items()}
    train_df.sort_values(by=['Camera', 'time_segment_id'], inplace=True)

    # --- PHASE 1: ENTER PREDICTION ---
    print("\n--- Training ENTER ---")
    train_df['target_temp'] = train_df['congestion_enter_rating'].map(label_map)
    train_df['target_future'] = train_df.groupby('Camera')['target_temp'].shift(-SHIFT)
    
    train_clean = train_df.dropna(subset=['target_future']).copy()
    y_enter = train_clean['target_future'].astype(int)
    X_enter = train_clean[feature_cols]
    
    imputer = SimpleImputer(strategy='median')
    X_enter_imp = imputer.fit_transform(X_enter)
    X_test_imp = imputer.transform(test_df[feature_cols])

    model_enter = get_stable_ensemble()
    model_enter.fit(X_enter_imp, y_enter)
    
    # Get Probabilities instead of hard classes
    enter_probs = model_enter.predict_proba(X_test_imp)
    # Apply Thresholding logic
    enter_preds_idx = apply_thresholds(enter_probs, label_map)

    # --- PHASE 2: EXIT PREDICTION (CHAINED) ---
    print("\n--- Training EXIT (Chained) ---")
    train_df['target_temp_exit'] = train_df['congestion_exit_rating'].map(label_map)
    train_df['target_future_exit'] = train_df.groupby('Camera')['target_temp_exit'].shift(-SHIFT)
    train_clean_exit = train_df.dropna(subset=['target_future_exit']).copy()
    
    # Add ENTER Prediction as a feature for EXIT
    # This is the "Chain Link"
    X_exit = train_clean_exit[feature_cols].copy()
    X_exit['enter_rating_feature'] = train_clean_exit.groupby('Camera')['target_temp'].shift(-SHIFT)
    
    # Re-impute
    X_exit_imp = imputer.fit_transform(X_exit)
    y_exit = train_clean_exit['target_future_exit'].astype(int)
    
    # Add Predicted Enter to Test
    X_test_exit = pd.DataFrame(X_test_imp, columns=feature_cols)
    X_test_exit['enter_rating_feature'] = enter_preds_idx
    
    # Use a fresh imputer for safety
    imputer_exit = SimpleImputer(strategy='median')
    X_exit_imp = imputer_exit.fit_transform(X_exit)
    X_test_exit_imp = imputer_exit.transform(X_test_exit)

    model_exit = get_stable_ensemble()
    model_exit.fit(X_exit_imp, y_exit)
    
    # Predict Exit
    exit_probs = model_exit.predict_proba(X_test_exit_imp)
    exit_preds_idx = apply_thresholds(exit_probs, label_map)

    # --- 3. SAVE ---
    print("--- Saving ---")
    test_df['predicted_id'] = test_df['time_segment_id'] + SHIFT
    preds_map = {}
    
    ent_str = [reverse_map[p] for p in enter_preds_idx]
    ext_str = [reverse_map[p] for p in exit_preds_idx]

    for i, row in test_df.iterrows():
        eid = int(row['predicted_id'])
        cam = row['Camera']
        preds_map[f"time_segment_{eid}_{cam}_congestion_enter_rating"] = ent_str[i]
        preds_map[f"time_segment_{eid}_{cam}_congestion_exit_rating"] = ext_str[i]

    final_rows = []
    for rid in sample_sub['ID'].values:
        pred = preds_map.get(rid, "free flowing")
        final_rows.append({"ID": rid, "Target": pred, "Target_Accuracy": pred})
        
    pd.DataFrame(final_rows).to_csv('submission_infinity.csv', index=False)
    print("SUCCESS: 'submission_infinity.csv' created.")

if __name__ == "__main__":
    main()