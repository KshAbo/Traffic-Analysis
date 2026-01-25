import cv2
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from computer_vision.tracker_ultralytics import UltralyticsTracker
from computer_vision.roi_loader import load_rois

from computer_vision.attributes.counts import VehicleCountAggregator
from computer_vision.attributes.flow import FlowAggregator
from computer_vision.attributes.density import DensityAggregator
from computer_vision.attributes.motion import MotionAggregator
from computer_vision.attributes.dwell import DwellTimeAggregator
from computer_vision.attributes.entry_exit_delay import EntryExitDelayAggregator
from computer_vision.attributes.vehicle_composition import VehicleTypeCompositionAggregator
from computer_vision.attributes.aggregator import MinuteAggregator


# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = "data/dataset/test"
ROI_CONFIG = "src/config.json"
TRAIN_CSV = "data/TestInputSegments.csv"
OUTPUT_CSV = "outputs/test_features.csv"

YOLO_MODEL = "yolov8x.pt"
CONF_THRESH = 0.3

FRAME_SAMPLING_FPS = 2
MINUTE_SECONDS = 60


# -------------------------
# LOAD METADATA
# -------------------------
train_df = pd.read_csv(TRAIN_CSV)
rois = load_rois(ROI_CONFIG)

Path("outputs").mkdir(exist_ok=True)

csv_file = open(OUTPUT_CSV, "w", newline="")
csv_writer = None


# -------------------------
# MAIN LOOP OVER DATASET
# -------------------------
for idx, row in tqdm(
                    train_df.iterrows(),
                    total=len(train_df),
                    desc="Videos"
                ):

    cam_idx = row["videos"].split('/')[0][-1]
    camera_id = "camera_" + cam_idx
    video_name = row["videos"].split('/')[-1]

    video_path = f"{DATASET_DIR}/normanniles{cam_idx}/{video_name}"

    if not Path(video_path).exists():
        print(f"[WARN] Missing video: {video_path}")
        continue

    # -------------------------
    # VIDEO SETUP
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    sampling_stride = int(video_fps / FRAME_SAMPLING_FPS)
    frames_per_minute = FRAME_SAMPLING_FPS * MINUTE_SECONDS

    # -------------------------
    # ROIs
    # -------------------------
    entry_roi = rois[camera_id]["entry"]
    exit_roi = rois[camera_id]["exit"]
    queue_roi = entry_roi

    # -------------------------
    # RESET TRACKER + AGGS
    # -------------------------
    tracker = UltralyticsTracker(
        model_path=YOLO_MODEL,
        conf=CONF_THRESH
    )

    aggregator = MinuteAggregator(
        VehicleCountAggregator(),
        FlowAggregator(entry_roi, exit_roi),
        DensityAggregator(queue_roi),
        MotionAggregator(),
        DwellTimeAggregator(queue_roi),
        EntryExitDelayAggregator(entry_roi, exit_roi),
        VehicleTypeCompositionAggregator(queue_roi)
    )

    frame_idx = 0
    processed_frames = 0
    minute_idx = 0

    # -------------------------
    # FRAME LOOP
    # -------------------------
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sampled_frames = total_frames // sampling_stride

    with tqdm(
        total=total_sampled_frames,
        desc=f"Frames ({video_name})",
        leave=False
    ) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sampling_stride != 0:
                frame_idx += 1
                continue

            tracks = tracker.track(frame)
            aggregator.update(tracks)

            processed_frames += 1

            if processed_frames == frames_per_minute:
                features = aggregator.finalize_minute()

                features["camera_id"] = camera_id
                features["video_file"] = video_name
                features["minute"] = minute_idx

                if csv_writer is None:
                    csv_writer = csv.DictWriter(
                        csv_file,
                        fieldnames=features.keys()
                    )
                    csv_writer.writeheader()

                csv_writer.writerow(features)
                csv_file.flush()

                minute_idx += 1
                processed_frames = 0

            frame_idx += 1

    cap.release()


csv_file.close()
print(f"\nFinished. Features written to {OUTPUT_CSV}")
