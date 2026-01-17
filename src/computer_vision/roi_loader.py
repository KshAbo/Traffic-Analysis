import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
print(ROOT)

from src.utils.roi_selector import ROI


def load_rois(json_path):
    '''
    How to use
    rois = load_rois("config/roi.json")
    ENTRY_ROI = rois["camera_1"]["entry"]
    EXIT_ROI  = rois["camera_1"]["exit"]
    '''
    with open(json_path, "r") as f:
        cfg = json.load(f)

    rois = {}
    for cam, cam_rois in cfg["roi"].items():
        rois[cam] = {
            "entry": ROI(*cam_rois["roi_entry"]),
            "exit": ROI(*cam_rois["roi_exit"])
        }

    return rois
