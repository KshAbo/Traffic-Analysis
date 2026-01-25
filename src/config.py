# src/config.py

ID_COL = "time_segment_id"

TARGET_ENTER = "congestion_enter_rating"
TARGET_EXIT  = "congestion_exit_rating"

TRAIN_PATH = "../data/Train.csv"
TEST_PATH  = "../data/TestInputSegments.csv"
CV_PATH = "../data/cv_features_train_ready.csv"

SUBMISSION_PATH_ENTER = "../submissions/submission_enter.csv"
SUBMISSION_PATH_EXIT  = "../submissions/submission_exit.csv"

ROLLING_WINDOW_MINUTES = 15
EMBARGO_MINUTES = 2
FORECAST_HORIZON_MINUTES = 5

TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
