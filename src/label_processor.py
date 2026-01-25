# src/label_processor.py

import pandas as pd
from config import (
    TARGET_ENTER,
    TARGET_EXIT,
    EMBARGO_MINUTES,
    FORECAST_HORIZON_MINUTES
)

LABEL_MAP = {
    "free flowing": 0,
    "light delay": 1,
    "moderate delay": 2,
    "heavy delay": 3
}

SHIFT = EMBARGO_MINUTES + FORECAST_HORIZON_MINUTES


def encode_and_shift_targets(df):
    df = df.copy()

    # Encode labels
    df["enter_encoded"] = df[TARGET_ENTER].map(LABEL_MAP)
    df["exit_encoded"]  = df[TARGET_EXIT].map(LABEL_MAP)

    assert df["enter_encoded"].isna().sum() == 0
    assert df["exit_encoded"].isna().sum() == 0

    # Shift targets forward (forecast)
    df["enter_target"] = df["enter_encoded"].shift(-SHIFT)
    df["exit_target"]  = df["exit_encoded"].shift(-SHIFT)

    # Drop rows where future target is unknown
    df = df.dropna().reset_index(drop=True)

    return df
