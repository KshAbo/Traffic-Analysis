import pandas as pd
import numpy as np
import warnings
import os
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import minimize
from sklearn.utils import resample
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SHIFT = 7 
SEED = 42
PSEUDO_LABEL_THRESHOLD = 0.94  

# --- 1. DATA TOOLS ---
def augment_data(df, target_col, multiplier=1):
    """ STRATEGY A: For Enter Model (Keep all data, duplicate rare) """
    rare_3 = df[df[target_col] == 3]
    rare_2 = df[df[target_col] == 2]
    
    dfs = [df]
    for _ in range(multiplier * 3): dfs.append(rare_3)
    for _ in range(multiplier): dfs.append(rare_2)
        
    return pd.concat(dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)

def get_balanced_dataset(df, target_col, target_size=3000):
    """ STRATEGY B: For Exit Model (Force 1:1 ratio, throw away majority) """
    df_0 = df[df[target_col] == 0]
    df_1 = df[df[target_col] == 1]
    df_2 = df[df[target_col] == 2]
    df_3 = df[df[target_col] == 3]
    
    if len(df_3) == 0: df_3 = df_0.iloc[:1] 
    
    df_0_res = resample(df_0, replace=False, n_samples=target_size, random_state=SEED)
    df_1_res = resample(df_1, replace=True,  n_samples=target_size, random_state=SEED)
    df_2_res = resample(df_2, replace=True,  n_samples=target_size, random_state=SEED)
    df_3_res = resample(df_3, replace=True,  n_samples=target_size, random_state=SEED)
    
    return pd.concat([df_0_res, df_1_res, df_2_res, df_3_res]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# --- 2. FEATURE ENGINEERING ---
def create_global_features(df):
    agg_dict = {'mean_vehicle_count': ['mean', 'max'], 'jam_factor': ['mean', 'max']}
    valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    if valid_agg:
        global_stats = df.groupby('time_segment_id').agg(valid_agg)
        global_stats.columns = ['_'.join(col).strip() + '_global' for col in global_stats.columns.values]
        df = df.merge(global_stats, on='time_segment_id', how='left')
    return df

def extract_time_from_filename(filename):
    # Extracts YYYY-MM-DD-HH-MM-SS from "normanniles1_2025-10-20-06-00-45.mp4"
    try:
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', str(filename))
        if match:
            return pd.to_datetime(match.group(1), format='%Y-%m-%d-%H-%M-%S')
    except:
        pass
    return pd.NaT

def feature_engineering(df):
    # 1. TIME FEATURES (CRITICAL FIX)
    print("  Extracting Time info from filenames...")
    
    # Try to get 'video_file' column or similar
    if 'video_file' in df.columns:
        dt_series = df['video_file'].apply(extract_time_from_filename)
    elif 'filename' in df.columns:
        dt_series = df['filename'].apply(extract_time_from_filename)
    else:
        # Fallback if we absolutely can't find it (shouldn't happen with your files)
        dt_series = pd.to_datetime(df['time_segment_id'], unit='m', origin='2025-01-01') 

    # Fill missing with a default (noon) to prevent crashes
    dt_series = dt_series.fillna(pd.Timestamp('2025-01-01 12:00:00'))

    # Extract Components
    df['hour'] = dt_series.dt.hour
    df['minute_of_day'] = dt_series.dt.hour * 60 + dt_series.dt.minute
    df['day_of_week'] = dt_series.dt.dayofweek # 0=Mon, 6=Sun
    
    # Cyclical Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
    df['hour_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
    
    # TRAFFIC LOGIC FLAGS
    # Morning Rush: 7am - 9am
    df['is_morning_rush'] = df['hour'].apply(lambda h: 1 if 7 <= h <= 9 else 0)
    # Evening Rush: 4pm - 6pm
    df['is_evening_rush'] = df['hour'].apply(lambda h: 1 if 16 <= h <= 18 else 0)
    # Night: 10pm - 5am
    df['is_night'] = df['hour'].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    # Weekend
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)

    # 2. PHYSICS
    if 'entry_count' in df.columns and 'exit_count' in df.columns:
        df['pressure'] = df['entry_count'] - df['exit_count']
    else: df['pressure'] = 0
        
    if 'mean_vehicle_count' in df.columns and 'stop_ratio' in df.columns:
        df['jam_factor'] = df['mean_vehicle_count'] * (df['stop_ratio'] + 0.001)
    else: df['jam_factor'] = 0

    cam_map = {"Norman Niles #1": 1, "Norman Niles #2": 2, "Norman Niles #3": 3, "Norman Niles #4": 4}
    if 'Camera' in df.columns: df['camera_code'] = df['Camera'].map(cam_map).fillna(0)

    # 3. ROLLING STATS
    df = df.sort_values(by=['Camera', 'time_segment_id'])
    cols_to_roll = [c for c in ['mean_vehicle_count', 'jam_factor', 'pressure'] if c in df.columns]
    if cols_to_roll:
        grp = df.groupby('Camera')
        for col in cols_to_roll:
            df[f'{col}_roll_5'] = grp[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df[f'{col}_accel'] = df[col] - df[f'{col}_roll_5']
            
            # Interaction: Is pressure rising DURING rush hour?
            df[f'{col}_rush_interaction'] = df[f'{col}_accel'] * (df['is_morning_rush'] + df['is_evening_rush'])

    return df

# --- 3. MODEL & OPTIMIZER ---
def get_classifier_ensemble(weights=None):
    cw = weights if weights else 'balanced'
    xgb_clf = xgb.XGBClassifier(n_estimators=1200, max_depth=7, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, eval_metric='mlogloss')
    lgb_clf = lgb.LGBMClassifier(n_estimators=1200, num_leaves=35, learning_rate=0.02, class_weight=cw, n_jobs=-1, random_state=SEED, verbose=-1)
    rf_clf = RandomForestClassifier(n_estimators=600, max_depth=12, min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)
    return VotingClassifier(estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('rf', rf_clf)], voting='soft', weights=[2.0, 1.5, 1.0])

class ProbOptimizer:
    def __init__(self): self.coef_ = [1.0, 1.0, 1.0, 1.0]
    def _loss_func(self, coef, probs, y_true): return -f1_score(y_true, np.argmax(probs * coef, axis=1), average='macro')
    def fit(self, probs, y_true): self.coef_ = minimize(self._loss_func, [1]*4, args=(probs, y_true), method='Nelder-Mead', tol=1e-2).x
    def predict(self, probs): return np.argmax(probs * self.coef_, axis=1)

# --- 4. MAIN PIPELINE ---
def main():
    print("--- 1. LOADING ---")
    base_dirs = ["../data/", "data/", "./"]
    files = {'train_feats': 'cv_features_train_ready.csv', 'test_feats': 'cv_features_test_ready.csv', 'train_labels': 'Train.csv', 'test_meta': 'TestInputSegments.csv', 'sample_sub': 'SampleSubmission.csv'}
    loaded = {}
    for key, filename in files.items():
        for base in base_dirs:
            fp = os.path.join(base, filename)
            if os.path.exists(fp):
                loaded[key] = pd.read_csv(fp)
                break
        if key not in loaded: return print(f"ERROR: {filename} not found.")

    train_feats, test_feats = loaded['train_feats'], loaded['test_feats']
    train_labels, test_meta = loaded['train_labels'], loaded['test_meta']
    sample_sub = loaded['sample_sub']

    if 'view_label' in train_labels.columns: train_labels.rename(columns={'view_label': 'Camera'}, inplace=True)
    if 'view_label' in test_meta.columns: test_meta.rename(columns={'view_label': 'Camera'}, inplace=True)

    train_df = pd.merge(train_feats, train_labels, on=['time_segment_id', 'Camera'], how='inner')
    test_df = pd.merge(test_feats, test_meta, on=['time_segment_id', 'Camera'], how='left')

    print("--- 2. ENGINEERING ---")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    train_df = create_global_features(train_df)
    test_df = create_global_features(test_df)

    exclude = ['camera_id', 'video_file', 'clean_filename', 'Camera', 'congestion_enter_rating', 'congestion_exit_rating', 'target_future', 'time_segment_id', 'minute', 'ID', 'dt', 'video_time', 'videos', 'date', 'datetimestamp_start', 'datetimestamp_end', 'responseId', 'ID_enter', 'ID_exit', 'signaling', 'cycle_phase', 'target_temp', 'enter_encoded', 'exit_encoded', 'enter_target', 'exit_target', 'target_temp_exit', 'target_future_exit']
    feature_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
    print(f"Features: {len(feature_cols)}")

    label_map = {"free flowing": 0, "light delay": 1, "moderate delay": 2, "heavy delay": 3}
    reverse_map = {v: k for k, v in label_map.items()}
    train_df.sort_values(by=['Camera', 'time_segment_id'], inplace=True)

    # --- PHASE 1: ENTER PREDICTION (Hybrid) ---
    print("\n--- Training ENTER ---")
    train_df['target_temp'] = train_df['congestion_enter_rating'].map(label_map)
    train_df['target_future'] = train_df.groupby('Camera')['target_temp'].shift(-SHIFT)
    train_clean = train_df.dropna(subset=['target_future']).copy()
    
    X = train_clean[feature_cols]
    y = train_clean['target_future'].astype(int)
    
    # Validation Split
    split_idx = int(len(train_clean) * 0.85)
    X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train on Augment
    train_aug = augment_data(pd.concat([X_tr, y_tr], axis=1), 'target_future', multiplier=2)
    model_enter = get_classifier_ensemble('balanced')
    model_enter.fit(train_aug[feature_cols].fillna(0), train_aug['target_future'].astype(int))

    # Pseudo-Labeling (Fixed Leakage)
    print("  Applying Pseudo-Labeling...")
    test_probs_initial = model_enter.predict_proba(test_df[feature_cols].fillna(0))
    high_conf_idx = np.where(np.max(test_probs_initial, axis=1) > PSEUDO_LABEL_THRESHOLD)[0]
    
    if len(high_conf_idx) > 0:
        pseudo_X = test_df.iloc[high_conf_idx][feature_cols]
        pseudo_y = np.argmax(test_probs_initial[high_conf_idx], axis=1)
        pseudo_df = pseudo_X.copy(); pseudo_df['target_future'] = pseudo_y
        
        # Combine X_tr (NOT X_val) with Pseudo
        train_combined = pd.concat([pd.concat([X_tr, y_tr], axis=1), pseudo_df])
        train_final = augment_data(train_combined, 'target_future', multiplier=2)
        model_enter.fit(train_final[feature_cols].fillna(0), train_final['target_future'].astype(int))
    
    # Validation Stats
    val_probs = model_enter.predict_proba(X_val.fillna(0))
    opt_enter = ProbOptimizer()
    opt_enter.fit(val_probs, y_val)
    val_preds = opt_enter.predict(val_probs)
    f1_ent = f1_score(y_val, val_preds, average='macro')
    acc_ent = accuracy_score(y_val, val_preds)
    zindi_ent = (0.7 * f1_ent) + (0.3 * acc_ent)

    # Final Prediction
    # For final model, combine ALL train data + pseudo
    if len(high_conf_idx) > 0:
        full_train = pd.concat([train_clean[feature_cols + ['target_future']], pseudo_df])
    else:
        full_train = train_clean[feature_cols + ['target_future']]
    
    full_aug = augment_data(full_train, 'target_future', multiplier=2)
    model_enter.fit(full_aug[feature_cols].fillna(0), full_aug['target_future'].astype(int))
    
    test_probs_ent = model_enter.predict_proba(test_df[feature_cols].fillna(0))
    enter_preds_final = np.argmax(test_probs_ent, axis=1)

    # --- PHASE 2: EXIT PREDICTION (Balanced) ---
    print("\n--- Training EXIT ---")
    train_df['target_temp_exit'] = train_df['congestion_exit_rating'].map(label_map)
    train_df['target_future_exit'] = train_df.groupby('Camera')['target_temp_exit'].shift(-SHIFT)
    train_clean_exit = train_df.dropna(subset=['target_future_exit']).copy()
    
    X_exit = train_clean_exit[feature_cols].copy()
    X_exit['enter_rating_feature'] = train_clean_exit['target_future']
    y_exit = train_clean_exit['target_future_exit'].astype(int)

    split_idx = int(len(X_exit) * 0.85)
    X_tr_ex, X_val_ex = X_exit.iloc[:split_idx], X_exit.iloc[split_idx:]
    y_tr_ex, y_val_ex = y_exit.iloc[:split_idx], y_exit.iloc[split_idx:]

    train_bal_ex = get_balanced_dataset(pd.concat([X_tr_ex, y_tr_ex], axis=1), 'target_future_exit', target_size=3000)
    model_exit = get_classifier_ensemble(weights=None)
    model_exit.fit(train_bal_ex.drop(columns=['target_future_exit']).fillna(0), train_bal_ex['target_future_exit'].astype(int))

    # Pseudo
    X_test_exit = test_df[feature_cols].copy()
    X_test_exit['enter_rating_feature'] = enter_preds_final 
    
    print("  Applying Pseudo-Labeling...")
    test_probs_initial_ex = model_exit.predict_proba(X_test_exit.fillna(0))
    high_conf_idx_ex = np.where(np.max(test_probs_initial_ex, axis=1) > 0.96)[0]
    
    if len(high_conf_idx_ex) > 0:
        pseudo_X_ex = X_test_exit.iloc[high_conf_idx_ex]
        pseudo_y_ex = np.argmax(test_probs_initial_ex[high_conf_idx_ex], axis=1)
        pseudo_df_ex = pseudo_X_ex.copy(); pseudo_df_ex['target_future_exit'] = pseudo_y_ex
        
        # Combine X_tr + Pseudo
        combined_raw = pd.concat([pd.concat([X_tr_ex, y_tr_ex], axis=1), pseudo_df_ex])
        train_final_ex = get_balanced_dataset(combined_raw, 'target_future_exit', target_size=3500)
        model_exit.fit(train_final_ex.drop(columns=['target_future_exit']).fillna(0), train_final_ex['target_future_exit'].astype(int))

    # Stats
    val_probs_ex = model_exit.predict_proba(X_val_ex.fillna(0))
    opt_exit = ProbOptimizer()
    opt_exit.fit(val_probs_ex, y_val_ex)
    val_preds_ex = opt_exit.predict(val_probs_ex)
    f1_ext = f1_score(y_val_ex, val_preds_ex, average='macro')
    acc_ext = accuracy_score(y_val_ex, val_preds_ex)
    zindi_ext = (0.7 * f1_ext) + (0.3 * acc_ext)

    # Final Pred
    if len(high_conf_idx_ex) > 0:
         # Need to reconstruct full dataset properly using the FULL data, not just X_tr
         # For simplicity in this script, we re-use X_exit (the clean train set) + Pseudo
         full_raw = pd.concat([pd.concat([X_exit, y_exit], axis=1), pseudo_df_ex])
    else:
         full_raw = pd.concat([X_exit, y_exit], axis=1)
         
    full_bal_ex = get_balanced_dataset(full_raw, 'target_future_exit', target_size=3500)
    model_exit.fit(full_bal_ex.drop(columns=['target_future_exit']).fillna(0), full_bal_ex['target_future_exit'].astype(int))
    
    test_probs_ext = model_exit.predict_proba(X_test_exit.fillna(0))
    exit_preds_final = []
    heavy_idx = label_map["heavy delay"]
    for row in test_probs_ext:
        if row[heavy_idx] > 0.35: exit_preds_final.append(heavy_idx)
        else: exit_preds_final.append(np.argmax(row))

    # --- REPORT ---
    print("\n" + "="*40)
    print("      FINAL PERFORMANCE REPORT      ")
    print("="*40)
    print(f"ENTER MODEL:")
    print(f"  Macro F1:     {f1_ent:.4f}")
    print(f"  Accuracy:     {acc_ent:.4f}")
    print(f"  Zindi Score:  {zindi_ent:.4f}")
    print("-" * 20)
    print(f"EXIT MODEL:")
    print(f"  Macro F1:     {f1_ext:.4f}")
    print(f"  Accuracy:     {acc_ext:.4f}")
    print(f"  Zindi Score:  {zindi_ext:.4f}")
    print("="*40)
    print(f"OVERALL ESTIMATED SCORE: {(zindi_ent + zindi_ext) / 2:.4f}")
    print("="*40 + "\n")

    # --- SAVE ---
    print("--- Saving ---")
    test_df['predicted_id'] = test_df['time_segment_id'] + SHIFT
    preds_map = {}
    ent_str = [reverse_map[p] for p in enter_preds_final]
    ext_str = [reverse_map[p] for p in exit_preds_final]

    for i, row in test_df.iterrows():
        eid = int(row['predicted_id'])
        cam = row['Camera']
        preds_map[f"time_segment_{eid}_{cam}_congestion_enter_rating"] = ent_str[i]
        preds_map[f"time_segment_{eid}_{cam}_congestion_exit_rating"] = ext_str[i]

    final_rows = []
    ids_to_predict = sample_sub['ID'].values if sample_sub is not None else preds_map.keys()
    for rid in ids_to_predict:
        pred = preds_map.get(rid, "free flowing")
        final_rows.append({"ID": rid, "Target": pred, "Target_Accuracy": pred})
        
    pd.DataFrame(final_rows).to_csv('submission_time_fixed.csv', index=False)
    print("SUCCESS: 'submission_time_fixed.csv' created.")

if __name__ == "__main__":
    main()