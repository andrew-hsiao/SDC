import numpy as np
import pickle
import cv2
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from vd.feat_ext import FeatureExtracter

import logging
logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)

class Partitioner:
    def _partition(self, width, height, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = width
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = height
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def __call__(self, width, height):
        windows = []
        #windows.append(self._partition(width, height, xy_window = (96, 72), x_start_stop=[7, None], y_start_stop=[3+int(height/2), int(height/2+72*2)], xy_overlap=(0.5, 0.5)))
        windows.append(self._partition(width, height, xy_window = (72, 56), x_start_stop=[None, None], y_start_stop=[int(height/2)+24, int(height/2+72*2)], xy_overlap=(0.75, 0.5)))
        windows.append(self._partition(width, height, xy_window = (132, 96), x_start_stop=[7, None], y_start_stop=[int(height/2), int(height/2+96*2)], xy_overlap=(0.6, 0.5)))
        windows.append(self._partition(width, height, xy_window = (192, 176), x_start_stop=[17, None], y_start_stop=[int(height*2/3), int(height*2/3+96*2)], xy_overlap=(0.5, 0.5)))

        #windows.append(self._partition(width, height, xy_window = (96, 96), y_start_stop=[300, None]))
        #windows.append(self._partition(width, height, xy_window = (128, 128), x_start_stop=[36, None],y_start_stop=[300, 600], xy_overlap=(0.85, 0.5)))
        #windows.append(self._partition(width, height, xy_window = (160, 160), x_start_stop=[36, None],y_start_stop=[360, 360+160], xy_overlap=(0.85, 0.5)))
        #windows.append(self._partition(width, height, xy_window = (160, 160), y_start_stop=[440, None]))
        #windows.append(self._partition(width, height, xy_window = (192, 192), y_start_stop=[440, None]))
        #windows.append(self._partition(width, height, xy_window = (256, 256), y_start_stop=[600, None]))
        #windows.append(self._partition(width, height, xy_window = (512, 512), y_start_stop=[600, None]))
        windows = [win for group in windows for win in group]
        return windows

class VehicleDetectAgent(object):
    MODEL_FILENAME = "model.p"
    def __init__(self, thres_cls = 0.9, thres_heat = 1):
        super().__init__()
        self.model = svm.SVC(kernel='linear', C=0.1, probability=True)
        self.best_model = self.model
        self.scaler = StandardScaler()
        self.threshold_cls = thres_cls
        self.threshold_heatmap = thres_heat

    def _search_param(self, states, policies):
        #the best C is 0.1 in model selection
        params = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}]
        grid_searcher = GridSearchCV(self.model, params, cv=5)
        grid_searcher.fit(states, policies)
        self.best_model = grid_searcher.best_estimator_
        logger.warning("best C is: {0}".format(grid_searcher.cv_results_['params'][grid_searcher.best_index_]))
        self.model = self.best_model

    def load(self):
        with open(VehicleDetectAgent.MODEL_FILENAME, 'rb') as f:
            pickle_data = pickle.load(f)
            self.model = pickle_data['model']
            self.scaler = pickle_data['scaler']

    def save(self):
        with open(VehicleDetectAgent.MODEL_FILENAME, 'wb') as f:
            pickle_data = {}
            pickle_data['model'] = self.model
            pickle_data['scaler'] = self.scaler
            pickle.dump(pickle_data, f)

    def normalize(self, features):
        return self.scaler.transform(features)

    def extract_feature(self, perception):
        spatial_size = (32,32)
        hist_bins = 32
        features = []
        labels = []
        for (s, a, r) in perception:
            hog_features = []
            img_features = []
            img=s
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            hog_features.extend(FeatureExtracter._hog(feature_image[:,:,0], orient=8, pix_per_cell=8, cell_per_block=2))
            hog_features.extend(FeatureExtracter._hog(feature_image[:,:,1], orient=8, pix_per_cell=8, cell_per_block=2))
            hog_features.extend(FeatureExtracter._hog(feature_image[:,:,2], orient=8, pix_per_cell=8, cell_per_block=2))
            img_features.append(hog_features)
            spatial_features = FeatureExtracter._bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
            hist_features = FeatureExtracter._color_hist(feature_image, nbins=hist_bins)
            img_features.append(hist_features)
            features.append(np.concatenate(img_features))
            labels.append(a)

        features = np.array(features, dtype = np.float64)
        labels = np.array(labels)
        return (features, labels)

    def imitate(self, perception):
        features, labels = self.extract_feature(perception)
        self.scaler = self.scaler.fit(features)
        self.model.fit(self.normalize(features), labels)

    def evaluate(self, states, policies):
        return self.model.score(states, policies)

    def action_classify(self, state):
        features, _ = self.extract_feature([(state, 0, 0)])
        prediction = self.model.predict_proba(self.normalize(features))
        if prediction[0][1] > self.threshold_cls:
            return 1
        else:
            return 0

    def action_detect(self, state):
        #input image, output action_id, bounding_boxes
        img = state
        part = Partitioner()
        height = img.shape[0]
        width = img.shape[1]
        regions = []
        perceptions = []
        windows = part(width, height)
        for window in windows:
            img_crop = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            logger.debug("VehicleDetectionEnv::generate_states() crop: {0}".format(img_crop.shape))
            perceptions.append((cv2.resize(img_crop, (64, 64)), 0, 0))

        features, _ = self.extract_feature(perceptions)
        prob = self.model.predict_proba(self.normalize(features))
        act = prob[:, 1] > self.threshold_cls
        return [windows[i] for i, _ in enumerate(windows) if act[i] == True ]

    def action_heatmap(self, state, windows):
        #input image, output single channel heatmap
        heatmap = np.zeros_like(state[:,:,0]).astype(np.float)
        for box in windows:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[heatmap <= self.threshold_heatmap] = 0
        heatmap = np.clip(heatmap, 0, 255)
        return heatmap

    def action(self, state):
        #input image, output region
        from scipy.ndimage.measurements import label
        windows = self.action_detect(state)
        heatmap = self.action_heatmap(state, windows)
        labels = label(heatmap)

        region = []
        for index in range(1, labels[1]+1):
            nonzero = (labels[0] == index).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            region.append(bbox)

        return region
