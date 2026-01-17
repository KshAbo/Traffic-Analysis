import cv2
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ROI:
    """
    Represents a rectangular Region of Interest (ROI)
    """
    x: int
    y: int
    w: int
    h: int

    def box(self) -> Tuple[int, int, int, int]:
        """
        Returns ROI as (x1, y1, x2, y2)
        """
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def crop(self, frame):
        """
        Crops the ROI from a frame
        """
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]

    def contains_point(self, px: int, py: int) -> bool:
        """
        Checks if a point lies inside the ROI
        """
        return (
            self.x <= px <= self.x + self.w and
            self.y <= py <= self.y + self.h
        )

    def contains_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Checks if the center of a bounding box lies inside the ROI
        bbox = (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self.contains_point(cx, cy)

    def draw(self, frame, color=(255, 0, 0), thickness=2, label=None):
        """
        Draws the ROI on a frame (for debugging / visualization)
        """
        x1, y1, x2, y2 = self.box()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if label:
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return frame
