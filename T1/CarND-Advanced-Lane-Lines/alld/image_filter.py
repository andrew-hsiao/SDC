import numpy as np
import cv2

import logging
logger = logging.getLogger(__name__)

class GradientFilter(object):
    '''GradientFilter implements the low level gradient functions

    API
        apply(image, type, kernel_size, thres)
        canny(image, thres)
    '''

    def __init__(self):
        super().__init__()

    def apply(self, image, type, kernel_size, thres):
        '''Apply filter on image

        There are 4 type filter:
        'x' filter: X directional filter
        'y' filter: Y directional filter
        'm' filter: X&Y (magnitude) filter
        'd' filter: Directional filter

        Args:
            image (numpy array)(HxW): single channel image (8-bit)
            type (character): one of 'x'/'y'/'m'/'d' to specify filter type
            kernel_size (int): filter size: {3,5,7,9}
            thres (tupple of two int): min/max threshold

        Returns:
            boolean values (numpy array)(HxW): activation is true if the
            gradient within threshold
        '''
        if type == 'x':
            sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobel = np.absolute(sx)
            #normalize
            sobel = np.uint8(sobel * 255 / np.max(sobel))
        elif type == 'y':
            sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
            sobel = np.absolute(sy)
            #normalize
            sobel = np.uint8(sobel * 255 / np.max(sobel))
        elif type == 'm':
            sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
            sobel = (sx**2+sy**2) ** 0.5
            #normalize
            sobel = np.uint8(sobel * 255 / np.max(sobel))
        elif type == 'd':
            sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
            sx = np.absolute(sx)
            sy = np.absolute(sy)
            sobel = np.arctan2(sy, sx)
        else:
            logger.error("%s Unknow filter type:%s", self, type)

        return ((sobel >= thres[0]) & (sobel <= thres[1]))

    def canny(self, image, thres):
        '''Apply canny filter to detect edge

        The function will create a new edge image

        Args:
            image (numpy array)(HxW): single channel image (8-bit)
            thres (tupple of two int): min/max threshold

        Returns:
            detected edge image
        '''
        return cv2.Canny(image, thres[0] , thres[1])

class EdgeDetector(object):
    '''EdgeDetector contains many selected combinations of image channel, type,
    parameter to detect edge in different images by GradientFilter
    '''

    def __init__(self):
        super().__init__()
        self.filter = GradientFilter()

    def detect(self, image):
        '''Detect edge for common case

        Args:
            image (numpy array)(HxWxC): RGB image

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        return self.detect_edge_complex_3(image)

    def detect_edge_complex_0(self, image):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls_h, hls_l, hls_s = cv2.split(hls)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ksize = 3
        thres = [(30, 180), (30, 180), (30, 180), (0.5, 1.5)]
        gx = self.filter.apply(gray, 'x', ksize, thres[0])
        gy = self.filter.apply(gray, 'y', ksize, thres[1])
        gm = self.filter.apply(gray, 'm', ksize, thres[2])
        gd = self.filter.apply(gray, 'd', ksize, thres[3])

        gh = ((hls_s >= 140) & (hls_s <= 255))

        img_binary = np.zeros_like(gray)
        grad_comb = (((gx | gy | gm) & gd) | gh)
        img_binary[grad_comb] = 1
        return img_binary

    def detect_edge_complex_1(self, img_in):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        luv = cv2.cvtColor(img_in, cv2.COLOR_RGB2Luv)
        luv_l, luv_u, luv_v = cv2.split(luv)
        hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)

        ksize = 3
        thres = [(10, 230), (10, 230), (20, 190), (0.7, 1.5)]

        img_channel = luv_v
        gx = self.filter.apply(img_channel, 'x', ksize, thres[0])
        gy = self.filter.apply(img_channel, 'y', ksize, thres[1])
        gm = self.filter.apply(img_channel, 'm', ksize, thres[2])
        gd = self.filter.apply(img_channel, 'd', ksize, thres[3])
        gcomb_1 = ((gx & gy) | (gm & gd))

        img_channel = luv_l
        gx = self.filter.apply(img_channel, 'x', ksize, thres[0])
        gy = self.filter.apply(img_channel, 'y', ksize, thres[1])
        gm = self.filter.apply(img_channel, 'm', ksize, thres[2])
        gd = self.filter.apply(img_channel, 'd', ksize, thres[3])
        gcomb_2 = (((gx | gy) |gm) & gd)

        img_channel = hsv_s
        gcomb_3 = self.filter.apply(img_channel, 'm', ksize, thres[2])

        img_channel = luv_u
        gcomb_4 = self.filter.apply(img_channel, 'm', ksize, thres[2])

        img_binary = np.zeros_like(img_channel)
        img_binary[((gcomb_1 | gcomb_2) | (gcomb_3 & gcomb_4))] = 1

        return img_binary

    def detect_edge_complex_2(self, img_in):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        luv = cv2.cvtColor(img_in, cv2.COLOR_RGB2Luv)
        luv_l, luv_u, luv_v = cv2.split(luv)
        hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)
        hls = cv2.cvtColor(img_in, cv2.COLOR_RGB2HLS)
        hls_h, hls_l, hls_s = cv2.split(hls)

        #M
        ksize = 3
        img_channel = hsv_h
        thres = [(80, 255), (80, 255), (80, 255), (0.5, 1.0)]
        gm_1 = self.filter.apply(img_channel, 'm', ksize, thres[2])

        thres = [(30, 80), (30, 80), (30, 80), (0.8, 1.3)]
        gx = self.filter.apply(img_channel, 'x', ksize, thres[0])
        gy = self.filter.apply(img_channel, 'y', ksize, thres[1])
        gm = self.filter.apply(img_channel, 'm', ksize, thres[2])
        gd = self.filter.apply(img_channel, 'd', ksize, thres[3])
        comb_1 = ((gx | gy | gm) & gd)

        #M
        thres = [(50, 120), (50, 120), (50, 120), (0.7, 1.3)]
        img_channel = hls_s
        gm_2 = self.filter.apply(img_channel, 'm', ksize, thres[2])

        #X&Y | M&D
        thres = [(30, 80), (30, 80), (30, 80), (0.8, 1.3)]
        img_channel = luv_v
        gx = self.filter.apply(img_channel, 'x', ksize, thres[0])
        gy = self.filter.apply(img_channel, 'y', ksize, thres[1])
        gm = self.filter.apply(img_channel, 'm', ksize, thres[2])
        gd = self.filter.apply(img_channel, 'd', ksize, thres[3])
        comb = (comb_1 | gm_1 | gm_2 | (gx & gy) | (gm & gd))

        img_binary = np.zeros_like(img_channel)
        img_binary[comb] = 1
        return img_binary

    def detect_edge_complex_3(self, image, ksize=3, thres=[(20, 130), (20, 130), (20, 130), (0.3, 1.2)]):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        return (self.detect_edge_gray_m_and_d(image, ksize, thres) |
            self.detect_edge_hlss_xoyom_and_d(image, ksize, thres))

    def detect_edge_hsvvm_or_luvvm(self, image, ksize, thres):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        luv = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        luv_l, luv_u, luv_v = cv2.split(luv)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)

        gm_hsv_v = edge_dector.filter.apply(hsv_v, 'm', ksize, thres[2])
        gm_luv_v = edge_dector.filter.apply(luv_v, 'm', ksize, thres[2])

        chn_edge_phv = np.zeros_like(luv_v)
        chn_edge_phv[(gm_hsv_v | gm_luv_v)] = 1
        return chn_edge_phv

    def detect_edge_gray_m_and_d(self, image, ksize, thres):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.edge_detect_m_and_d(gray, ksize, thres)

    def detect_edge_hlss_xoyom_and_d(self, image, ksize, thres):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls_h, hls_l, hls_s = cv2.split(hls)
        return self.edge_detect_xoyom_and_d(hls_s, ksize, thres)

    def detect_edge_hsvv_xoryorm_and_d(self, image, ksize, thres):
        '''Detect edge by complex combination

        Args:
            image (numpy array)(HxWxC): RGB image
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)
        return self.edge_detect_xoyom_and_d(hsv_v, ksize, thres)

    def edge_detect_x_and_y(self, chn, ksize, thres):
        '''Detect edge by gradient x and y from channel

        Args:
            chn (numpy array)(HxW): single channel image (8-bit)
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        gx = self.filter.apply(chn, 'x', ksize, thres[0])
        gy = self.filter.apply(chn, 'y', ksize, thres[1])
        chn_edge_phv = np.zeros_like(chn)
        chn_edge_phv[(gx & gy)] = 1
        return chn_edge_phv

    def edge_detect_xnm_and_ynm(self, chn, ksize, thres):
        '''Detect edge by gradient x and y from channel

        Args:
            chn (numpy array)(HxW): single channel image (8-bit)
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        gx = self.filter.apply(chn, 'x', ksize, thres[0])
        gy = self.filter.apply(chn, 'y', ksize, thres[1])
        gm = self.filter.apply(chn, 'm', ksize, thres[2])

        chn_edge_phv = np.zeros_like(chn)
        chn_edge_phv[(gx & gm) | ((gy & gm))] = 1
        return chn_edge_phv

    def edge_detect_xoyom_and_d(self, chn, ksize, thres):
        '''Detect edge by gradient x and y from channel

        Args:
            chn (numpy array)(HxW): single channel image (8-bit)
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        gx = self.filter.apply(chn, 'x', ksize, thres[0])
        gy = self.filter.apply(chn, 'y', ksize, thres[1])
        gm = self.filter.apply(chn, 'm', ksize, thres[2])
        gd = self.filter.apply(chn, 'd', ksize, thres[3])

        chn_edge_phv = np.zeros_like(chn)
        chn_edge_phv[(gx | gy | gm) & gd] = 1
        return chn_edge_phv

    def edge_detect_m_and_d(self, chn, ksize, thres):
        '''Detect edge by gradient x and y from channel

        Args:
            chn (numpy array)(HxW): single channel image (8-bit)
            ksize (int): Filter kernel size
            thres (4 tuples array): the threshold of x, y, m, d gradients

        Returns:
            image (numpy array)(HxW): detected edge image
        '''
        gm = self.filter.apply(chn, 'm', ksize, thres[2])
        gd = self.filter.apply(chn, 'd', ksize, thres[3])
        chn_edge_phv = np.zeros_like(chn)
        chn_edge_phv[gm & gd] = 1
        return chn_edge_phv

    def create_all_channels(self, image):
        '''Create a list of all the channels of images

        Args:
            image (numpy array)(HxWxC): RGB image

        Returns:
            image array (numpy array), name array (string array)

            images = [(gray, (gray_r, gray_g, gray_b)),
                        (hls, (hls_h, hls_l, hls_s)),
                        (luv, (luv_l, luv_u, luv_v)),
                        (hsv, (hsv_h, hsv_s, hsv_v))]
        '''
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        luv = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_r, gray_g, gray_b = cv2.split(image)
        hls_h, hls_l, hls_s = cv2.split(hls)
        luv_l, luv_u, luv_v = cv2.split(luv)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)

        gray = np.dstack((gray,gray,gray))*255

        images = [(gray, (gray_r, gray_g, gray_b)),
        (hls, (hls_h, hls_l, hls_s)),
        (luv, (luv_l, luv_u, luv_v)),
        (hsv, (hsv_h, hsv_s, hsv_v))]

        names = [('gray', ('gray_r', 'gray_g', 'gray_b')),
        ('hls', ('hls_h', 'hls_l', 'hls_s')),
        ('luv', ('luv_l', 'luv_u', 'luv_v')),
        ('hsv', ('hsv_h', 'hsv_s', 'hsv_v'))]

        return images, names

    def detect_hough_lines(self, image, rho, theta, thres, min_line_len, max_line_gap):
        '''Line detector

        Args:
            image (numpy array)(HxW): single channel image (8-bit)
            rho (int): distance of the line from original
            theta (int): angle of the line perpendicular to the detected line
            thres (int): the threshold value for image
            min_line_len (int): minimum length of line
            max_line_gap (int): maximum allowed gap between line segments to treat them as single line

        Returns:
            detected hough lines
        '''
        return cv2.HoughLinesP(image, rho, theta, thres, minLineLength=min_line_len, maxLineGap=max_line_gap)