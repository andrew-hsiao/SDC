import pickle
import glob
import cv2
import os
import numpy as np
import matplotlib.image as mpimg

import logging
logger = logging.getLogger(__name__)

class CameraAuxiliary(object):
    '''CameraAuxiliary is the major class for camera calibration

    API
        calibrate(image)
        fit(cb_fns, cb_size)
        draw_chessboard_corners(cb_image, cb_size)
        save(cal_fn)
    '''
    DEFAULT_CALBRIATION_FILENAME = "cam_cal.p"

    def _init_imp(self):
        '''Initialization'''
        super(CameraAuxiliary, self).__init__()
        self.image_points = []
        self.object_points = []
        self.camera_matrix = None
        self.dist_coeff = None
        self.image_size = None
        self.rotate_vec = None
        self.translate_vec = None

    def __init__(self, cal_fn = ""):
        '''Initialization by existing calibration file

        The function will load calibration data through cache file

        Args:
            cal_fn (string): a pickle file name of calibration data
        '''
        super().__init__()
        self._init_imp()
        if os.path.exists(cal_fn):
            self._load(cal_fn)
            logger.info("Camera calibration file loaded:", cal_fn)
        else:
            logger.info("No default camera calibration file:", cal_fn)

    def calibrate(self, image):
        '''Calibrates the image

        The function will calibrate image throgh undistortion

        Args:
            image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)

        Returns:
            imaeg (HxWxC)(C:RGB): calibrated image
        '''
        if image.shape != self.image_size:
            self.image_size = image.shape[0:2]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, self.image_size, None, None)

            self.camera_matrix = mtx
            self.dist_coeff = dist
            self.rotate_vec = rvecs
            self.translate_vec = tvecs

        return cv2.undistort(image, self.camera_matrix, self.dist_coeff)

    def fit(self, cb_fns, cb_size):
        '''Fit chessboard images to calculate image points

        Args:
            cb_fns (list of string): list of chessboard images filename
            cb_size (tuple of int): (column, row) of chessboard
        '''
        self.image_points, self.object_points, cb_images = self._gen_cb_corners(cb_fns, cb_size)
        return cb_images

    def draw_chessboard_corners(self, cb_image, cb_size):
        '''Draw chessboard corners on image

        Args:
            cb_image (HxWxC)(C:RGB): the chessboard image
            cb_size (tuple of int): (column, row) of chessboard
        '''
        gray = cv2.cvtColor(cb_image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, cb_size)
        if ret == True:
            cb_image = cv2.drawChessboardCorners(cb_image, cb_size, corners, True)
        else:
            logger.warning("%s: cv2.drawChessboardCorners() fail.", self)

        return cb_image

    def save(self, cal_fn):
        '''Save calibration data into file

        Args:
            cal_fn (string): target file name
        '''
        with open(cal_fn, 'wb') as f:
            pickle_data = {}
            pickle_data['object_points'] = self.object_points
            pickle_data['image_points'] = self.image_points
            pickle.dump(pickle_data, f)

    def _load(self, cal_fn):
        with open(cal_fn, 'rb') as f:
            pickle_data = pickle.load(f)
            self.image_points = pickle_data['image_points']
            self.object_points = pickle_data['object_points']

    def _gen_cb_grid(self, cb_size):
        #mesh grid (54,2): (0,0), (1,0), (2,0)...(8,0), (0,1), (1,1)...(8.5)
        #return (r,c,0) by mgrid[0:c, 0:r].T
        col, row = cb_size
        grid = np.zeros((col*row, 3), np.float32)
        grid[:,:2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
        return grid

    def _gen_cb_corners(self, cb_fns, cb_size):
        images = []
        cb_corners = []
        cb_points = []
        cb_images = []

        for cb_fn in cb_fns:
            cb_img = mpimg.imread(cb_fn)
            gray = cv2.cvtColor(cb_img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, cb_size)
            cb_size_result = np.copy(cb_size)
            if ret != True:
                #search possible fit by: (x-1,y), (x-2, y), (x,y-1), (x,y-2), (x-1, y-2)
                #reduce_corners = [(-1,0), (-2,0), (0,-1), (0,-2), (-1,-1), (-1,-2), (-2,-1)]
                reduce_corners = np.mgrid[-3:1, -3:1].T.reshape(-1, 2)
                for rc in reduce_corners:
                    cb_size_new = np.copy(cb_size)
                    cb_size_new = tuple(cb_size_new + rc)
                    #print("try:",cb_size_new)
                    logger.debug("try:(%d,%d)",cb_size_new)
                    ret, corners = cv2.findChessboardCorners(gray, cb_size_new)
                    if ret == True:
                        break

                if ret != True:
                    logger.warning("%s: cv2.findChessboardCorners() fail. fn:%s", self, cb_fn)
                    #give up the image
                    continue

                #print("try successed")
                cb_size_result = cb_size_new

            grid = self._gen_cb_grid(cb_size_result)
            cb_points.append(grid)
            cb_corners.append(corners)
            cb_images.append(cb_img)

        return cb_corners, cb_points, cb_images
