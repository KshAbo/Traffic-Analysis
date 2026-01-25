# src/data_loader.py

import pandas as pd
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_ENTER, TARGET_EXIT

def load_and_aggregate():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    train = train.sort_values(by="time_segment_id").reset_index(drop=True)
    test = test.sort_values(by="time_segment_id").reset_index(drop=True)
    return train, test
