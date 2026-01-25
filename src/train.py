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
from feature_builder import merge_cv_features

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
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=y
    )

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
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"{target_name} Validation Accuracy: {acc:.4f}")
    
    return model

def main():
    # -----------------------
    # 1. Load Data (Labels)
    # -----------------------
    print("Loading base data...")
    # NOTE: Ensure data_loader.py is fixed to NOT group by ID.
    train_df, test_df = load_and_aggregate() 

    # -----------------------
    # 2. Load & Merge CV Features
    # -----------------------
    print(f"Loading CV features from {CV_PATH}...")
    try:
        cv_features = pd.read_csv(CV_PATH)
        
        # Merge features into Train and Test
        train_df = merge_cv_features(train_df, cv_features)
        test_df = merge_cv_features(test_df, cv_features)
        print("Feature merge successful.")
        
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find {CV_PATH}.")
        print("Make sure your teammate's file is in the data folder.")
        return

    # -----------------------
    # 3. Process Labels (Encoding + Shift)
    # -----------------------
    print("Processing labels and creating forecast lags...")
    # This function now handles the shift for forecasting
    train_df = encode_and_shift_targets(train_df)

    # -----------------------
    # 4. Prepare Feature Matrix (X)
    # -----------------------
    # Exclude ID and Target columns from the feature set
    ignore_cols = [
        ID_COL, 
        "congestion_enter_rating", "congestion_exit_rating", # Original Text
        "enter_target", "exit_target",                       # Shifted Targets
        "enter_encoded", "exit_encoded",                     # Raw Encoded
        "target", "Target", "Target_Accuracy"                # Junk columns
    ]
    
    # Select all columns that are NOT in the ignore list
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    print(f"Training on {len(feature_cols)} features: {feature_cols}")

    X = train_df[feature_cols]
    
    # Targets
    y_enter = train_df["enter_target"].astype(int)
    y_exit  = train_df["exit_target"].astype(int)

    # -----------------------
    # 5. Train Models
    # -----------------------
    model_enter = train_single_model(X, y_enter, "ENTER RATING")
    model_exit  = train_single_model(X, y_exit,  "EXIT RATING")

    # -----------------------
    # 6. Generate Predictions on Test Set
    # -----------------------
    print("\nGenerating predictions for Test Set...")
    
    # Ensure Test has the same features
    X_test = test_df[feature_cols]
    
    pred_enter = model_enter.predict(X_test)
    # pred_exit = model_exit.predict(X_test) # Calculate this too if needed later

    # -----------------------
    # 7. Create Submission File
    # -----------------------
    submission = pd.DataFrame({
        "ID": test_df[ID_COL],
        "Target": [REVERSE_LABEL_MAP[p] for p in pred_enter],
        "Target_Accuracy": [REVERSE_LABEL_MAP[p] for p in pred_enter] # Duplicate col as per sample
    })
    
    # Save
    output_path = "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission.head())

if __name__ == "__main__":
    main()