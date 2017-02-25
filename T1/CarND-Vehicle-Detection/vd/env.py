from vd.feat_ext import FeatureExtracter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import cv2
import matplotlib.image as mpimg
import numpy as np
import logging
import numpy as np
from numpy.random import randint
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VehicleDetectionEnv(object):
    ACTION_NOT_CAR = 0
    ACTION_CAR = 1
    ACTION_UNKNOWN = 2
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.dataset = {}

    def attach(self, filenames, action = ACTION_UNKNOWN):
        if action not in self.dataset.keys():
            self.dataset[action] = []

        self.dataset[action].extend(filenames)

    #return percept: (state, action, reward)
    def observation(self):
        for action, filenames in self.dataset.items():
            for fn in filenames:
                img = mpimg.imread(fn)
                if img.dtype == np.float32:
                    img = (img*255).astype(np.uint8)
                yield (img, action, 1)

    def state(self, action = VehicleDetectionEnv.ACTION_UNKNOWN, index = -1):
        if index == -1:
            index = randint(len(self.dataset[action][index]))

        fn = self.dataset[action][index]
        img = mpimg.imread(fn)
        if img.dtype == np.float32:
            img = (img*255).astype(np.uint8)
        return img
