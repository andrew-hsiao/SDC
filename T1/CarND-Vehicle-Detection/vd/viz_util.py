import numpy as np
import cv2
import logging
logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)

class VisualUtil:
    @staticmethod
    def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
        imcopy = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy
