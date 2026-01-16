import cv2
from pathlib import Path
import sys
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from src.computer_vision.detector import VehicleDetector

# ----------------------------
# CONFIG
# ----------------------------
INPUT_VIDEO = str(Path.joinpath(PROJECT_ROOT, "data", "dataset", "normanniles1", "normanniles1_2025-10-20-06-00-45.mp4"))
OUTPUT_VIDEO = PROJECT_ROOT / "outputs" / "sample_detected.mp4"
CONF_THRESH = 0.3

# ----------------------------
# SETUP
# ----------------------------
Path("outputs").mkdir(exist_ok=True)
print(Path.exists(Path(INPUT_VIDEO)))

detector = VehicleDetector(conf=CONF_THRESH)

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_idx = 0

# ----------------------------
# MAIN LOOP
# ----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    writer.write(frame)

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx} frames")

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
writer.release()

print(f"Saved output video to: {OUTPUT_VIDEO}")
