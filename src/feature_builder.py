# src/feature_builder.py

import pandas as pd
from config import (
    ID_COL,
    ROLLING_WINDOW_MINUTES
)

def merge_cv_features(base_df, cv_df):
    """
    base_df : one row per time_segment_id
    cv_df   : one row per time_segment_id (from teammate)
    """
    assert ID_COL in cv_df.columns

    merged = base_df.merge(cv_df, on=ID_COL, how="left")

    # Ensure no missing CV rows
    assert merged.isna().sum().sum() == 0

    return merged


def add_rolling_features(df, feature_cols):
    df = df.sort_values(ID_COL).reset_index(drop=True)

    for col in feature_cols:
        df[f"{col}_mean_15"] = df[col].rolling(ROLLING_WINDOW_MINUTES).mean()
        df[f"{col}_std_15"]  = df[col].rolling(ROLLING_WINDOW_MINUTES).std()
        df[f"{col}_max_15"]  = df[col].rolling(ROLLING_WINDOW_MINUTES).max()

    # Drop initial rows where rolling window is incomplete
    df = df.dropna().reset_index(drop=True)

    return df
