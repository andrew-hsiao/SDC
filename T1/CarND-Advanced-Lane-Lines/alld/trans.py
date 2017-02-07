import numpy as np
import cv2

import logging
logger = logging.getLogger(__name__)

class PerspectiveTrans(object):
    '''The class will handle perspective transformation
    '''
    def __init__(self, img_size = (720, 1280), src = None, dst = None):
        '''top-left, bottom-left, bottom-right, top-right and  in (W, H) fashion
        '''
        super().__init__()
        h, w = img_size
        if src == None:
            self.src = np.float32(
                [[(w / 2) - 64, h / 2 + 100],
                [((w / 6) - 10), h],
                [(w * 5 / 6) + 60, h],
                [(w / 2 + 70), h / 2 + 100]])

        if dst == None:
            self.dst = np.float32(
                [[(w / 4), 0],
                [(w / 4), h],
                [(w * 3 / 4), h],
                [(w * 3 / 4), 0]])

        self.trans_matrix = cv2.getPerspectiveTransform(self.src, self.dst)

    def warp(self, image):
        '''
            Args:
                image (numpy array)(HxWXC): image for perspective transformation

            Returns:
                image (numpy array)(HxWXC): image after transformed

        '''
        return cv2.warpPerspective(image, self.trans_matrix, (image.shape[1], image.shape[0]))

    def dewarp(self, image):
        '''
            Args:
                image (numpy array)(HxWXC): image for perspective transformation

            Returns:
                image (numpy array)(HxWXC): image after transformed

        '''
        return cv2.warpPerspective(image, np.linalg.inv(self.trans_matrix), (image.shape[1], image.shape[0]))
