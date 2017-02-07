import numpy as np
import collections

from collections import deque

import logging
logger = logging.getLogger(__name__)

class LaneLine(object):
    '''LaneLine manages a N degrees polynomial curve which would be
    a set of points (x, y)
    '''
    def __init__(self, dim, meter_per_pixel, degree = 2, mem_frames = 30*18):
        '''Initialization function
        Args:
            height (int): the height
            meter_per_pixel (tupple of float): x_meter_per_pixel, y_meter_per_pixel
        '''
        super().__init__()
        self.degree = degree

        #self.fit: coefficients of polynomial curve
        self.meter_per_pixel = meter_per_pixel
        self.dim = dim
        self.x_vals = deque(maxlen=60)
        self.y_vals = deque(maxlen=60)

        self.fit_curves = deque(maxlen=mem_frames)
        self.fit_lines = deque(maxlen=mem_frames)

        #perfect fitting by model
        self.curve = None
        self.line = None

    def fit(self, x_vals, y_vals):
        '''Fitting N degree polynomial coefficients by sorts of x, y values
        Args:
            x_vals (numpy array): array of int of x axis
            y_vals (numpy array): array of int of y axis
        '''
        width = self.dim[1]
        self.x_vals.append(x_vals)
        self.y_vals.append(y_vals)

        coef, residule, rank, sing, rcond = np.polyfit(y_vals, x_vals, self.degree, full=True)
        if len(residule) == 0:
            rmse = 0
        else:
            rmse = (residule[0]/len(x_vals))**.5
        self.fit_curves.append((coef, rmse))

        coef, residule, rank, sing, rcond = np.polyfit(y_vals, x_vals, 1, full=True)
        if len(residule) == 0:
            mse = 0
        else:
            rmse = (residule[0]/len(x_vals))**.5
        self.fit_lines.append((coef, rmse))

        self.curve = None
        self.line = None

    def rollback_fit(self, step):
        '''rollback previous fit result it will be used when want to apply another scan
        Args:
            step (int): how many steps want to rollback
        '''
        for i in range(step):
            self.x_vals.pop()
            self.y_vals.pop()
            self.fit_lines.pop()
            self.fit_curves.pop()

    def fitted_curve(self):
        '''return the fitted curve
        '''
        if self.curve is not None:
            return self.curve

        height = self.dim[0]
        y_vals = np.linspace(0, height-1, height)
        x_vals = self.predict(y_vals, True)
        self.curve = np.vstack([x_vals, y_vals]).T
        return self.curve

    def fitted_line(self):
        '''return the fitted line
        '''
        if self.line is not None:
            return self.line

        height = self.dim[0]
        y_vals = np.linspace(0, height-1, height)
        x_vals = self.predict(y_vals, False)
        self.line = np.vstack([x_vals, y_vals]).T
        return self.line

    def predict(self, y_vals, is_curve=True):
        '''Return x values by y values according to fitting result
        '''
        if is_curve == True:
            model = self.fit_curves[-1][0]
        else:
            model = self.fit_lines[-1][0]

        return np.polyval(model, y_vals)

    def radius_in_meter(self):
        '''get the radius in meter
        '''

        #x=f(y), roc = ((1+(dx/dy)^2)^(3/2)) / (d^2x/dy^2)
        #x=f(y) = a*y^2 + b*y^1 + c*y^0
        #dx/dy = 2*a*y + b, d^2x/dy^2 = 2*a
        x_vals = self.x_vals[-1]
        y_vals = self.y_vals[-1]
        height = self.dim[0]

        mpp_x = self.meter_per_pixel[0]
        mpp_y = self.meter_per_pixel[1]

        x_vals_m = x_vals * mpp_x
        y_vals_m = y_vals * mpp_y
        height_m = height * mpp_y

        coef = np.polyfit(y_vals_m, x_vals_m, 2)
        y_vals_m = np.linspace(0, height_m-1, height_m)
        x_vals_m = coef[0]*y_vals_m**2 + coef[1]*y_vals_m + coef[2]

        d1 = 2*coef[0]*height_m + coef[1]
        d2 = np.absolute(2*coef[0])
        roc_m = ((1+d1**2)**1.5) / d2
        return roc_m

    def offset_in_meter(self):
        '''get offset from center in meter
        '''

        mpp_x = self.meter_per_pixel[0]
        height = self.dim[0]
        width = self.dim[1]
        center = width/2
        x = self.predict([height])[0]
        offset = x - center
        return offset * mpp_x

    def slope(self):
        '''get the slop of fitted line
        '''
        #last, coef, coef[0]
        return self.fit_lines[-1][0][0]

    def cost(self, is_curve=True):
        '''get the cost of fit result
        '''
        if is_curve == True:
            cost = self.fit_curves[-1][1]
        else:
            cost = self.fit_lines[-1][1]
        return cost

    def derivative(self, is_curve=True, order=1):
        '''get the derivative of fitted curve/line
        '''

        #dx/dy = 2*a*y + b, d^2x/dy^2 = 2*a
        if is_curve == True:
            model = self.fit_curves[-1][0]
        else:
            model = self.fit_lines[-1][0]

        return np.polyder(model,order)


class RoadManager(object):
    '''Road manager is the major class to manage lane lines
    API:
        detect(self, image)
        confidence(self)
        def detect_counter(self)
        def radius_in_meter(self)
        offset_in_meter(self)
        is_sraight(self)
    '''
    STRAIGHT_LINE_SLOPE_THRESHOLD = 0.05
    def __init__(self, dim, condidence_thres = 0.8, lane_width_in_pixels=600, scan_win_counts = 9, scan_win_width = 200):
        super().__init__()
        self.left_lane = None
        self.right_lane = None
        self.left_scan_wins = []
        self.right_scan_wins = []
        self.dim = dim
        self.margin = 200
        height = dim[0]
        self.left_lane = LaneLine(dim, (3.7/height, 30.0/height))
        self.right_lane = LaneLine(dim, (3.7/height, 30.0/height))
        self.conf = deque(maxlen=30*60)
        self.conf_fast = deque(maxlen=30*60)
        self.det_n = 0
        self.conf_thres = condidence_thres
        self.lane_width = lane_width_in_pixels
        self.scan_win_width = scan_win_width
        self.scan_win_counts = scan_win_counts

    def detect(self, image):
        '''the entry point to lane line detection
        Note:
            the function will try:
                1. scan_fast:
                2. scan:
                3. special scan (todo)
        Args:
            image (HxW)image (HxW): numpy array image with dimension (height, width, 3)image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)(C:RGB): numpy array image with dimension (height, width)

        Returns:
            confidence (0-1) of the detection
        '''
        #return (self.left_lane, self.right_lane)
        conf = 0

        if self.det_n != 0:
            conf = self._scan_fast(image)
            if conf < self.conf_thres:
                logger.warning("rescan in low confidence:{0}".format(conf))
                self.rollback_scan()
                #try _scan()
                conf = self._scan(image)
        else:
            conf = self._scan(image)

        self.conf.append(conf)
        self.det_n += 1
        return conf

    def confidence(self):
        '''confidence of the fitting result

            Note:
                cost (normalized MSE) of curve fitting
                distance of curvature within lanes
                similarity of curvature slopes within lanes
        '''
        width = self.dim[1]
        cost_left = self.left_lane.cost()
        cost_right = self.right_lane.cost()
        mean_cost = (cost_left + cost_right)/self.margin

        x_vals_left = self.left_lane.fitted_curve()[:,0]
        x_vals_right = self.right_lane.fitted_curve()[:,0]
        delta_dist = (x_vals_right - x_vals_left)
        mean_dist = np.mean(delta_dist)
        #0 is perfect, 1 is the worst, < 0.3 is good
        mean_dist = np.absolute(mean_dist - self.lane_width)/self.lane_width
        conf_mean_dist = 1-mean_dist
        if conf_mean_dist < 0:
            conf_mean_dist = 0

        #deriv_left = self.left_lane.derivative()
        #deriv_right = self.right_lane.derivative()
        #mean_curve = ((deriv_left - deriv_right) ** 2).mean() ** .5
        #delta_curves = np.absolute((np.arctan(deriv_left[1]) - np.arctan(deriv_right[1])))
        #print(delta_curves)

        conf = 0.5*(1-mean_cost) + 0.5*(conf_mean_dist)# + 0.2*(1-mean_curve)

        if conf < 0:
            logger.warning("Confidence < 0: cost:{0}, distance:{1}".format(mean_cost, mean_dist))
            conf = 0
        return conf

    @property
    def detect_counter(self):
        return self.det_n

    def radius_in_meter(self):
        return (self.left_lane.radius_in_meter() + self.right_lane.radius_in_meter())/2

    def offset_in_meter(self):
        return (self.left_lane.offset_in_meter() + self.right_lane.offset_in_meter())/2

    def is_sraight(self):
        return ((np.absolute(self.left_lane.slope()) < RoadManager.STRAIGHT_LINE_SLOPE_THRESHOLD) &
                    (np.absolute(self.right_lane.slope()) < RoadManager.STRAIGHT_LINE_SLOPE_THRESHOLD))

    def _scan(self, image):
        '''scan image by window

        Args:
            image (numpy array)(HxW): single channel image (8-bit) in bird-eyes view
            win_n (int): window counts for scanning image
            win_w (int): window width for scanning image

        Returns:
            tupple of scannig windows (left_top (x,y), right_bottom (x,y)) list
        '''
        img_in = image

        win_n = self.scan_win_counts
        win_w = self.scan_win_width

        significant_points_count = win_w / 8
        img_size = img_in.shape
        img_h = img_size[0]
        img_w = img_size[1]
        win_h = img_h//win_n
        win_m = win_w//2
        img_center_x = img_w//2
        img_center_y = img_h//2

        hist_bottom = np.sum(img_in[img_center_y:,:], axis=0)
        lane_left_x = np.argmax(hist_bottom[:img_center_x])
        lane_right_x = img_center_x + np.argmax(hist_bottom[img_center_x:])

        nonz_left_inds_x_vec = []
        nonz_left_inds_y_vec = []
        nonz_right_inds_x_vec = []
        nonz_right_inds_y_vec = []

        left_wins = []
        right_wins = []

        skip_scan_l = False
        skip_scan_r = False
        for win_i in range(win_n):
            #print(lane_left_x, lane_right_x)
            win_left_l = lane_left_x - win_m
            win_left_r = lane_left_x + win_m
            win_left_t = img_h - ((win_i+1) *  win_h)
            win_left_b = img_h - (win_i *  win_h)
            win_right_l = lane_right_x - win_m
            win_right_r = lane_right_x + win_m
            win_right_t = win_left_t
            win_right_b = win_left_b
            #next x position of sliding window is decided by the mean of nonzero points within previous window
            nonz = img_in[win_left_t:win_left_b+1, win_left_l:win_left_r+1].nonzero()
            nonz_left_inds_x = nonz[1]
            nonz_left_inds_x += win_left_l
            nonz_left_inds_y = nonz[0]
            nonz_left_inds_y += win_left_t

            nonz = img_in[win_right_t:win_right_b+1, win_right_l:win_right_r+1].nonzero()
            nonz_right_inds_x = nonz[1]
            nonz_right_inds_x += win_right_l
            nonz_right_inds_y = nonz[0]
            nonz_right_inds_y += win_right_t

            if (len(nonz_left_inds_x) > significant_points_count) and (skip_scan_l != True):
                nonz_left_inds_x_vec.append(nonz_left_inds_x)
                nonz_left_inds_y_vec.append(nonz_left_inds_y)
                lane_left_x = np.int(np.mean(nonz_left_inds_x))
                left_wins.append([(win_left_l, win_left_t), (win_left_r, win_left_b)])
            else:
                #skip scan
                #skip_scan_l = True
                pass

            if (len(nonz_right_inds_x) > significant_points_count) and (skip_scan_r != True):
                nonz_right_inds_y_vec.append(nonz_right_inds_y)
                nonz_right_inds_x_vec.append(nonz_right_inds_x)
                lane_right_x = np.int(np.mean(nonz_right_inds_x))
                right_wins.append([(win_right_l, win_right_t), (win_right_r, win_right_b)])
            else:
                #skip_scan_r = True
                pass

        #0 fit
        if(len(nonz_left_inds_x_vec) == 0  or len(nonz_right_inds_x_vec) == 0):
            err_msg = "Scan error: pix_x_l:{0}, pix_y_l:{1}, pix_x_r:{2}, pix_y_r:{3},".format(
            len(nonz_left_inds_x),
            len(nonz_left_inds_y),
            len(nonz_right_inds_x),
            len(nonz_right_inds_y))
            logger.warn(err_msg)
            return 0
        else:
            left_lane_x_inds = np.concatenate(nonz_left_inds_x_vec)
            left_lane_y_inds = np.concatenate(nonz_left_inds_y_vec)
            right_lane_x_inds = np.concatenate(nonz_right_inds_x_vec)
            right_lane_y_inds = np.concatenate(nonz_right_inds_y_vec)

        #print(len(left_lane_x_inds), len(left_lane_y_inds))
        #print(len(right_lane_x_inds), len(right_lane_y_inds))
        self.left_lane.fit(left_lane_x_inds, left_lane_y_inds)
        self.right_lane.fit(right_lane_x_inds, right_lane_y_inds)

        y_space = np.linspace(0, img_h-1, img_h)
        self.left_lane.predict(y_space)
        self.right_lane.predict(y_space)
        self.margin = win_w
        self.left_scan_wins = left_wins
        self.right_scan_wins = right_wins
        conf = self.confidence()
        return conf

    def _scan_fast(self, image):
        '''scan by previous fitted coefficients
        It will be more fast than rescan by sliding window.
        The assumption is current curve is simular with previous fitted

        Args:
            image (numpy array)(HxW): single channel image (8-bit) in bird-eyes view
            margin (int): margin to select pixels for scanning
        '''
        if(len(self.left_lane.fit_curves) == 0  or len(self.right_lane.fit_curves) == 0):
            self.conf_fast.append(0)
            return 0

        vals = image.nonzero()
        height = self.dim[0]
        y_vals = vals[0]
        x_vals = vals[1]

        x_vals_fitted_left = self.left_lane.predict(y_vals)
        x_vals_fitted_right = self.right_lane.predict(y_vals)

        margin = self.margin/2

        cond_left = ((x_vals > (x_vals_fitted_left-margin)) & (x_vals < (x_vals_fitted_left+margin)))
        cond_right = ((x_vals > (x_vals_fitted_right-margin)) & (x_vals < (x_vals_fitted_right+margin)))

        self.left_lane.fit(x_vals[cond_left], y_vals[cond_left])
        self.right_lane.fit(x_vals[cond_right], y_vals[cond_right])

        y_space = np.linspace(0, height-1, height)
        self.left_lane.predict(y_space)
        self.right_lane.predict(y_space)

        conf = self.confidence()
        self.conf_fast.append(conf)
        return conf

    def rollback_scan(self, step=1):
        self.left_lane.rollback_fit(step)
        self.right_lane.rollback_fit(step)
