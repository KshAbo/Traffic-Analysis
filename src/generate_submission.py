import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def main():
    print("--- 1. Loading Data ---")
    try:
        # Load the files you uploaded
        train_feats = pd.read_csv('../data/cv_features_train_ready.csv')
        test_feats = pd.read_csv('../data/cv_features_test_ready.csv')
        train_labels = pd.read_csv('../data/Train.csv')
        sample_sub = pd.read_csv('../data/SampleSubmission.csv')
    except FileNotFoundError:
        print("ERROR: Please ensure files are in the '../data/' folder.")
        # Fallback for current folder
        train_feats = pd.read_csv('cv_features_train_ready.csv')
        test_feats = pd.read_csv('cv_features_test_ready.csv')
        train_labels = pd.read_csv('Train.csv')
        sample_sub = pd.read_csv('SampleSubmission.csv')

    # Prepare Labels
    # Map 'view_label' -> 'Camera' to match features
    labels = train_labels[['time_segment_id', 'view_label', 'congestion_enter_rating']].copy()
    labels.rename(columns={'view_label': 'Camera'}, inplace=True)

    print("--- 2. Merging & Processing ---")
    train_df = pd.merge(train_feats, labels, on=['time_segment_id', 'Camera'], how='inner')
    
    # Encode Target
    label_map = {"free flowing": 0, "light delay": 1, "moderate delay": 2, "heavy delay": 3}
    reverse_map = {v: k for k, v in label_map.items()}
    train_df['target'] = train_df['congestion_enter_rating'].map(label_map)

    # --- THE GOLDEN KEY: +7 MINUTE SHIFT ---
    SHIFT = 7 
    train_df.sort_values(by=['Camera', 'time_segment_id'], inplace=True)
    train_df['target_future'] = train_df.groupby('Camera')['target'].shift(-SHIFT)
    train_clean = train_df.dropna(subset=['target_future'])

    # Select Features
    exclude = ['camera_id', 'video_file', 'clean_filename', 'Camera', 
               'congestion_enter_rating', 'target', 'target_future', 'time_segment_id', 'minute', 'ID']
    features = [c for c in train_clean.columns if c not in exclude and np.issubdtype(train_clean[c].dtype, np.number)]
    
    print(f"Training on {len(features)} features: {features}")

    # Train
    X = train_clean[features]
    y = train_clean['target_future'].astype(int)
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_imputed, y)

    # Predict
    print("--- 3. Predicting & Formatting ---")
    X_test = test_feats[features]
    X_test_imputed = imputer.transform(X_test)
    preds = clf.predict(X_test_imputed)
    
    test_feats['Target'] = [reverse_map[p] for p in preds]
    
    # Apply Shift to ID: Input T predicts T+7
    test_feats['predicted_id'] = test_feats['time_segment_id'] + SHIFT
    
    # Construct Zindi ID
    test_feats['ID'] = (
        "time_segment_" + 
        test_feats['predicted_id'].astype(str) + 
        "_" + 
        test_feats['Camera'] + 
        "_congestion_enter_rating"
    )
    
    # Final Merge with Sample Submission
    final_sub = pd.merge(sample_sub[['ID']], test_feats[['ID', 'Target']], on='ID', how='left')
    
    # Fill missing (edge cases) with default
    final_sub['Target'] = final_sub['Target'].fillna("free flowing")
    final_sub['Target_Accuracy'] = final_sub['Target']
    
    final_sub.to_csv('submission.csv', index=False)
    print("SUCCESS: Created 'submission.csv'. Upload this to Zindi!")

if __name__ == "__main__":
    main()