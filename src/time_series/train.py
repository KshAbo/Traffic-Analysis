import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    RANDOM_SEED,
    TEST_SPLIT_RATIO,
    ID_COL,
    CV_PATH,
    SUBMISSION_PATH_ENTER
)
from data_loader import load_and_aggregate
from label_processor import encode_and_shift_targets
from feature_builder import merge_cv_features, add_rolling_features

# Reverse mapping for Zindi
REVERSE_LABEL_MAP = {
    0: "free flowing",
    1: "light delay",
    2: "moderate delay",
    3: "heavy delay"
}

def train_single_model(X, y, target_name):
    """
    Trains a LightGBM classifier for a specific target (Enter or Exit).
    """
    print(f"\n--- Training Model for {target_name} ---")
    
    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=y
    )

    # Initialize Model
    # Note: We use 'multiclass' objective
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=30,
        random_state=RANDOM_SEED,
        verbose=-1
    )
    
    # Fit Model
    # REMOVED: early_stopping_rounds from .fit() to fix TypeError
    # We rely on callbacks for early stopping if needed, or just train full 1000 rounds (it's safe).
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=100)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks
    )
    
    # Validation Score
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"{target_name} Validation Accuracy: {acc:.4f}")
    
    return model

def main():
    # -----------------------
    # 1. Load Data
    # -----------------------
    train_df, test_df = load_and_aggregate()

    # -----------------------
    # 2. Load & Merge CV Features
    # -----------------------
    print(f"Loading CV features from {CV_PATH}...")
    try:
        cv_features = pd.read_csv(CV_PATH)
        train_df = merge_cv_features(train_df, cv_features)
        test_df = merge_cv_features(test_df, cv_features)
        print("Feature merge successful.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find {CV_PATH}.")
        return

    # -----------------------
    # 3. Feature Engineering (Rolling/Lags)
    # -----------------------
    # Define which columns to create stats for (only numeric ones)
    numeric_cols = [c for c in train_df.columns if train_df[c].dtype in ['float64', 'int64']]
    exclude_stats = ['time_segment_id', 'congestion_enter_rating', 'congestion_exit_rating', 'minute']
    cols_to_engineer = [c for c in numeric_cols if c not in exclude_stats]

    train_df = add_rolling_features(train_df, cols_to_engineer)
    test_df  = add_rolling_features(test_df, cols_to_engineer)

    # -----------------------
    # 4. Process Labels (Encoding + Shift)
    # -----------------------
    print("Processing labels and creating forecast lags...")
    train_df = encode_and_shift_targets(train_df)

    # -----------------------
    # 5. Prepare Feature Matrix (X)
    # -----------------------
    # CRITICAL FIX: Drop Non-Numeric Columns
    # We must remove IDs, Dates, Strings, and Target columns from X
    drop_cols = [
        ID_COL, 
        "congestion_enter_rating", "congestion_exit_rating", 
        "enter_target", "exit_target", 
        "enter_encoded", "exit_encoded", 
        "target", "Target", "Target_Accuracy",
        "responseId", "Camera", "ID_enter", "ID_exit", "videos", 
        "video_time", "datetimestamp_start", "datetimestamp_end", 
        "date", "signaling", "cycle_phase", "camera_id", "video_file", "clean_filename"
    ]
    
    # Select only columns that exist and are NOT in the drop list
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    # Extra safety: Select only numeric types
    X = train_df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()
    
    print(f"Training on {len(feature_cols)} numeric features: {feature_cols}")

    y_enter = train_df["enter_target"].astype(int)
    y_exit  = train_df["exit_target"].astype(int)

    # -----------------------
    # 6. Train Models
    # -----------------------
    model_enter = train_single_model(X, y_enter, "ENTER RATING")
    model_exit  = train_single_model(X, y_exit,  "EXIT RATING")

    # -----------------------
    # 7. Generate Predictions on Test Set
    # -----------------------
    print("\nGenerating predictions for Test Set...")
    
    # Ensure Test has same columns
    X_test = test_df[feature_cols].select_dtypes(include=[np.number])
    # Handle any missing columns (fill 0)
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols] # Ensure order
    
    pred_enter = model_enter.predict(X_test)
    pred_exit  = model_exit.predict(X_test) # We predict Exit too!

    # -----------------------
    # 8. Create Submission File
    # -----------------------
    # We need to construct the IDs for both Enter and Exit
    
    # ENTER Submission
    sub_enter = pd.DataFrame({
        "ID": test_df[ID_COL].astype(str) + "_" + test_df['Camera'] + "_congestion_enter_rating",
        "Target": [REVERSE_LABEL_MAP[p] for p in pred_enter]
    })
    
    # EXIT Submission (Zindi usually asks for one merged file, or check sample)
    # The sample provided only had Enter, but the problem asked for Exit.
    # We will generate a file that matches the SampleSubmission format for now.
    
    # IMPORTANT: The IDs in the SampleSubmission were likely NOT shifted.
    # But our Test Data IS the Input.
    # If we predicted T+7, we need to map to the correct ID.
    # Zindi IDs in SampleSubmission usually correspond to the TARGET time.
    # If the Sample IDs match our Test IDs + Shift, we are good.
    # Based on your previous success with +7, let's stick to the generated IDs.
    
    # Let's load the sample submission to see exactly which IDs are required
    try:
        sample_sub = pd.read_csv("../data/SampleSubmission.csv")
        required_ids = sample_sub['ID'].values
    except:
        required_ids = []

    # Create a dictionary of ID -> Prediction
    preds_dict = {}
    
    # Add ENTER predictions
    # Note: Shift logic for IDs
    SHIFT = 7 # From Config
    
    for i, row in test_df.iterrows():
        # Input Time + Shift = Target Time
        # Assuming time_segment_id is an integer we can add to
        target_segment_id = int(row[ID_COL]) + SHIFT
        
        # Construct ID
        enter_id = f"time_segment_{target_segment_id}_{row['Camera']}_congestion_enter_rating"
        exit_id  = f"time_segment_{target_segment_id}_{row['Camera']}_congestion_exit_rating"
        
        preds_dict[enter_id] = REVERSE_LABEL_MAP[pred_enter[i]]
        preds_dict[exit_id]  = REVERSE_LABEL_MAP[pred_exit[i]]

    # Map to Sample Submission
    final_rows = []
    if len(required_ids) > 0:
        for rid in required_ids:
            # Look up prediction, default to 'free flowing' if missing
            pred = preds_dict.get(rid, "free flowing")
            final_rows.append({"ID": rid, "Target": pred, "Target_Accuracy": pred})
        submission = pd.DataFrame(final_rows)
    else:
        # Fallback if no sample loaded
        submission = sub_enter
        submission['Target_Accuracy'] = submission['Target']

    output_path = "submission_lgbm.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()