import numpy as np
import cv2

from alld.cam import CameraAuxiliary
from alld.image_filter import EdgeDetector
from alld.trans import PerspectiveTrans
from alld.road_mgr import LaneLine, RoadManager

import logging
logger = logging.getLogger(__name__)

class VisualUtil(object):
    '''Visualization utility implements many function for visualization. It is a stateless class
    '''
    def __init__(self):
        super().__init__()

    def draw_selected_pixels(self, image, road, color=((255,0,255),(0,0,255)), alpha=0):
        '''draw the pixels be selected as a portion of lane with specified color

        Args:
            image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
            road (RoadManager): the instance of RoadManager
            color (2 RGB tuple): the colors of 2 lane lines
            alpha (float): 0-1 value for the alpha blending

        Returns:
            image (HxWxC)(C:RGB): output image
        '''
        image_blank = np.copy(image)
        left_lane = road.left_lane
        right_lane = road.right_lane
        #print(left_lane.y_vals[-1])
        #print(left_lane.x_vals[-1])
        image_blank[left_lane.y_vals[-1], left_lane.x_vals[-1]] = color[0]
        image_blank[right_lane.y_vals[-1], right_lane.x_vals[-1]] = color[1]
        if alpha != 0:
            return cv2.addWeighted(image, 1-alpha, image_blank, alpha, 0)
        else:
            return image_blank

    def draw_lane_line(self, image, road, color=(255,255,0), thickness=2, alpha=0):
        '''draw the lane line

        Args:
            image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
            road (RoadManager): the instance of RoadManager
            color (2 RGB tuple): the colors of 2 lane lines
            thickness (int): the thickness of line
            alpha (float): 0-1 value for the alpha blending

        Returns:
            image (HxWxC)(C:RGB): output image
        '''
        #image_blank = np.zeros_like(image)
        image_blank = np.copy(image)
        left_lane = road.left_lane
        right_lane = road.right_lane
        left_pts = left_lane.fitted_curve()
        right_pts = right_lane.fitted_curve()
        cv2.polylines(image_blank, np.int_([left_pts]), False, color, thickness)
        cv2.polylines(image_blank, np.int_([right_pts]), False, color, thickness)
        if alpha != 0:
            return cv2.addWeighted(image, 1-alpha, image_blank, alpha, 0)
        else:
            return image_blank

    def draw_lane_scan_windows(self, image, road, color=(255,255,0), thickness=2, alpha=0):
        '''draw the scan windows of lane

        Args:
            image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
            road (RoadManager): the instance of RoadManager
            color (2 RGB tuple): the colors of 2 lane lines
            thickness (int): the thickness of line
            alpha (float): 0-1 value for the alpha blending

        Returns:
            image (HxWxC)(C:RGB): output image
        '''
        image_blank = np.copy(image)

        for win in road.left_scan_wins:
            left_top = win[0]
            right_bottom = win[1]
            cv2.rectangle(image_blank, left_top, right_bottom, color, thickness)

        for win in road.right_scan_wins:
            left_top = win[0]
            right_bottom = win[1]
            cv2.rectangle(image_blank, left_top, right_bottom, color, thickness)

        if alpha != 0:
            return cv2.addWeighted(image, 1-alpha, image_blank, alpha, 0)
        else:
            return image_blank

    def draw_trans_box(self, image, pers_trans, color=[(255,0,0),(0,0,255)], thickness=2, alpha=0):
        '''draw the perspective transform box

        Args:
            image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
            pers_trans(PerspectiveTrans): the instance of PerspectiveTrans
            color (2 RGB tuple): the colors of 2 lane lines
            thickness (int): the thickness of line
            alpha (float): 0-1 value for the alpha blending

        Returns:
            image (HxWxC)(C:RGB): output image
        '''
        image_blank = np.copy(image)
        clr_1 = color[0]
        clr_2 = color[1]

        pts_src = np.array(pers_trans.src, np.int32)
        pts_src = pts_src.reshape((-1,1,2))
        cv2.polylines(image_blank,[pts_src],True,clr_1, thickness)

        pts_dst = np.array(pers_trans.dst, np.int32)
        pts_dst = pts_dst.reshape((-1,1,2))
        cv2.polylines(image_blank,[pts_dst],True,clr_2, thickness)

        if alpha != 0:
            return cv2.addWeighted(image, 1-alpha, image_blank, alpha, 0)
        else:
            return image_blank

    def draw_lane_area(self, image, road, color = (0, 255, 0), alpha=0.3):
        '''Fill the lane area within lines

        Args:
            image_dim (tupple of int)(H,W,C): image dimension
            left_lane (LineLane object): left lane
            right_lane (LineLane object): right lane

        Returns:
            image (numpy array)(HxWXC): output image with filled area
        '''
        image_blank = np.copy(image)

        left_lane = road.left_lane
        right_lane = road.right_lane

        left_pts = left_lane.fitted_curve()
        right_pts = np.flipud(right_lane.fitted_curve())
        pts = np.vstack((left_pts, right_pts))

        cv2.fillPoly(image_blank, np.int_([pts]), color)
        return cv2.addWeighted(image, 1-alpha, image_blank, alpha, 0)

    def draw_text(self, image, text, offset, font_size=2, color=(255,255,0), thickness=2):
        '''draw text on image

        Args:
            image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
            text (string): the string want to draw
            offset (x,y): the offset of left-top corner
            color (2 RGB tuple): the colors of 2 lane lines
            thickness (int): the thickness of line

        Returns:
            image (HxWxC)(C:RGB): output image
        '''
        image_blank = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_blank, text, offset, font, font_size, color, thickness)
        #return cv2.addWeighted(image, 1, image_blank, 0.9, 0)
        return image_blank

    def draw_pipeline_undistort_clr_phe(self, image, draw_box = False):
        '''draw pipeline: undistort with color by pinehole view

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
                draw_box (bool): draw the perspective transform box or not

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
        persp_trans = PerspectiveTrans()

        img_out = cam_aux.calibrate(image)
        if draw_box == True:
            img_out = self.draw_trans_box(img_out, persp_trans)

        return img_out

    def draw_pipeline_edge_bin_phe(self, image, draw_box = False):
        '''draw pipeline: edge with binary image by pinehole view

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
                draw_box (bool): draw the perspective transform box or not

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
        edge_dector = EdgeDetector()
        persp_trans = PerspectiveTrans()

        img_clr_phv_cal = cam_aux.calibrate(image)
        chn_edge_phv = edge_dector.detect(img_clr_phv_cal)
        img_out = np.dstack((chn_edge_phv,chn_edge_phv,chn_edge_phv))*255

        if draw_box == True:
            img_out = self.draw_trans_box(img_out, persp_trans)

        return img_out

    def draw_pipeline_trans_bin_bev(self, image, draw_box = False):
        '''draw pipeline: perspective transform with binary image by birds eye view

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
                draw_box (bool): draw the perspective transform box or not

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
        edge_dector = EdgeDetector()
        persp_trans = PerspectiveTrans()

        img_clr_phv_cal = cam_aux.calibrate(image)
        chn_bin_phv = edge_dector.detect(img_clr_phv_cal)
        img_bin_bev = np.dstack((chn_bin_phv,chn_bin_phv,chn_bin_phv))*255
        img_out = persp_trans.warp(img_bin_bev)

        if draw_box == True:
            img_out = self.draw_trans_box(img_out, persp_trans)

        return img_out

    def draw_pipeline_trans_clr_bev(self, image, draw_box = False):
        '''draw pipeline: perspective transform with color image by birds eye view

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
                draw_box (bool): draw the perspective transform box or not

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
        edge_dector = EdgeDetector()
        persp_trans = PerspectiveTrans()
        img_clr_phv_cal = cam_aux.calibrate(image)
        img_out = persp_trans.warp(img_clr_phv_cal)

        if draw_box == True:
            img_out = self.draw_trans_box(img_out, persp_trans)

        return img_out

    def draw_pipeline_lane_detection(self, image):
        '''draw pipeline: lane detection

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
        edge_dector = EdgeDetector()
        persp_trans = PerspectiveTrans()
        road = RoadManager((720,1280))
        img_clr_phv_cal = cam_aux.calibrate(image)
        chn_edge_phv = edge_dector.detect(img_clr_phv_cal)
        chn_edge_bev = persp_trans.warp(chn_edge_phv)
        road.detect(chn_edge_bev)

        img_bin_bev_raw = np.dstack((chn_edge_bev,chn_edge_bev,chn_edge_bev))*255
        img_bin_bev_det = self.draw_lane_scan_windows(img_bin_bev_raw, road)
        img_bin_bev_det = self.draw_lane_line(img_bin_bev_det, road, (0,255,0), 200, 0.2)
        img_bin_bev_det = self.draw_lane_line(img_bin_bev_det, road, (255,0,0), 2)
        img_bin_bev_det = self.draw_selected_pixels(img_bin_bev_det, road, ((255,0,0),(0,0,255)), 0.6)
        img_bin_bev_det = self.draw_lane_area(img_bin_bev_det, road, (0,255,0), 0.2)
        return img_bin_bev_det

    def draw_pipleline_final_image(self, image):
        '''draw pipeline: the final output

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
        edge_dector = EdgeDetector()
        persp_trans = PerspectiveTrans()
        road = RoadManager((720,1280))

        cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
        img_clr_phv_cal = cam_aux.calibrate(image)
        chn_edge_phv = edge_dector.detect(img_clr_phv_cal)
        chn_edge_bev = persp_trans.warp(chn_edge_phv)
        road.detect(chn_edge_bev)

        img_clr_phv_final = np.copy(img_clr_phv_cal)
        img_clr_bev_clean = np.zeros_like(img_clr_phv_final)
        img_clr_bev_clean = self.draw_lane_line(img_clr_bev_clean, road, (255,0,0), 5)
        img_clr_bev_clean = self.draw_selected_pixels(img_clr_bev_clean, road, ((0,255,255),(0,0,255)))
        img_clr_bev_clean = self.draw_lane_area(img_clr_bev_clean, road, (0,255,0))
        img_clr_bev_clean = persp_trans.dewarp(img_clr_bev_clean)
        img_clr_phv_final = cv2.addWeighted(img_clr_phv_final, 1, img_clr_bev_clean, 1, 0)

        if road.is_sraight():
            msg_roc = "Almost straight"
        else:
            msg_roc = "Radius of Curvature = {0:.2f}(m)".format(road.radius_in_meter())

        img_clr_phv_final = self.draw_text(img_clr_phv_final, msg_roc, (100, 100))
        offset = road.offset_in_meter()

        if offset > 0:
            shift_msg = "left"
        elif offset < 0:
            shift_msg = "right"
        else:
            shift_msg = "no"

        msg_offset = "Vechicle is {0:.2f}cm {1} shift".format(offset*100, shift_msg)
        img_clr_phv_final = self.draw_text(img_clr_phv_final, msg_offset, (100, 170))

        return img_clr_phv_final

    def tile_view_10(self, images):
        '''draw 10 images with tile view

            Args:
                images (array of 10 images): numpy array image with dimension (height, width, 3)

            Returns:
                image (HxWxC)(C:RGB): output image
        '''

        '''
            |03|04|
        1   ------|
            |05|06|
        ==========|
            |07|08|
        2   ------|
            |09|10|
        '''
        assert(len(images) == 10)
        dim = images[0].shape
        height = dim[0]
        width = dim[1]
        image_w_large = width//2
        image_h_large = height//2
        image_w_small = width//4
        image_h_small = height//4
        images_new = []
        dim_large = (image_h_large, image_w_large)
        dim_small = (image_h_small, image_w_small)
        dims = [dim_large,dim_large,dim_small,dim_small,dim_small,dim_small,dim_small,dim_small,dim_small,dim_small]
        pos = [(0,0), (image_h_large, 0),
        (0, image_w_large),
        (0, image_w_large+image_w_small),
        (image_h_small, image_w_large),
        (image_h_small, image_w_large+image_w_small),
        (image_h_large, image_w_large),
        (image_h_large, image_w_large+image_w_small),
        (image_h_large+image_h_small, image_w_large),
        (image_h_large+image_h_small, image_w_large+image_w_small)]

        image_blank = np.zeros_like(images[0])

        for idx in range(len(images)):
            #img_new = np.zeros(dims[idx])
            #print("image_blank:",image_blank.shape)
            d = dims[idx]
            p = pos[idx]
            #resize: (WxH)
            img_new = cv2.resize(images[idx], (d[1],d[0]))
            #print(img_new.shape, p, d)
            image_blank[p[0]:p[0]+d[0], p[1]:p[1]+d[1]] = img_new

        return image_blank

    def tile_view_16(self, images):
        '''draw 16 images with tile view

            Args:
                images (array of 16 images): numpy array image with dimension (height, width, 3)

            Returns:
                image (HxWxC)(C:RGB): output image
        '''

        '''
        01|02|03|04|
        --|--|--|--|
        05|06|07|08|
        --|--|--|--|
        09|10|11|12|
        --|--|--|--|
        13|14|15|16|
        '''
        assert(len(images) == 16)
        dim = images[0].shape
        height = dim[0]
        width = dim[1]
        pos = []
        dims = []
        grid_w = width//4
        grid_h = height//4
        for row in range(4):
            for col in range(4):
                dims.append((grid_h, grid_w))
                pos.append((row * grid_h, col * grid_w))

        image_blank = np.zeros_like(images[0])
        for idx in range(len(images)):
            d = dims[idx]
            p = pos[idx]
            #print(images[idx].shape, d, p)
            #resize: (WxH)
            img_new = cv2.resize(images[idx], (d[1],d[0]))
            #print(img_new.shape, p, d)
            image_blank[p[0]:p[0]+d[0], p[1]:p[1]+d[1]] = img_new

        return image_blank

    def blit_in_row(self, images):
        '''draw 4 images in a row

            Args:
                images (array of 10 images): numpy array image with dimension (height, width, 3)

            Returns:
                image (HxWxC)(C:RGB): output image
        '''

        images = self.chns_2_imgs(images)
        dims = np.array([img.shape for img in images])
        height_out = np.max(dims[:,0])
        width_out = np.max(dims[:,1]) * len(images)
        width_sum = np.sum(dims[:,1])
        gap_x = (width_out - width_sum) // len(images)
        dim_out = ((height_out, width_out, 3))
        image_blank = np.zeros(dim_out)

        pos_x = gap_x
        for img in images:
            pos_y = (img.shape[0] - height_out) // 2
            img_h = img.shape[0]
            img_w = img.shape[1]
            image_blank[pos_y:pos_y+img_h, pos_x:pos_x+img_w] = img*255
            #pos_y = pos_y + img_h + gap_y
            pos_x = pos_x + img_w + gap_x

        return image_blank

    def _fill_rect(self, image_dim, color = (0, 255, 0)):
        image_filled = np.zeros(image_dim, dtype=np.uint8)

        #y: (0,1,...,height-1) (N=height)
        y_val = np.linspace(0, image_dim[0]-1, image_dim[0])
        #x_l: (x_l, x_l, ..., x_l) (N=height)
        x_val_l = np.array([image_dim[1]//3]*image_dim[0])
        #x_r: (x_r, x_r, ..., x_r) (N=height)
        x_val_r = np.array([image_dim[1]*2//3]*image_dim[0])

        #pts_left: (x_l, 0), (x_l, 1), ...,(x_l, height-1) (Nx1)
        pts_left = np.vstack([x_val_l, y_val]).T
        #pts_right: (x_r, height-1), (x_r, height-2), ..., (x_r, 0) (Nx1)
        pts_right = np.flipud(np.vstack([x_val_r, y_val]).T)
        #pts: (x_l, 0), (x_l, 1), ...,(x_l, height-1), (x_r, height-1), (x_r, height-2), ..., (x_r, 0) (2Nx1)
        pts = np.vstack((pts_left, pts_right))
        cv2.fillPoly(image_filled, np.int_([pts]), color)
        return image_filled

    def chns_2_imgs(self, channels):
        return np.array([np.dstack((cn,cn,cn))*255 if len(cn.shape) < 3 else cn for cn in channels])

    def draw_all_channels(self, img):
        '''draw 16 channels of a image together

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        edge_dector = EdgeDetector()
        images, names = edge_dector.create_all_channels(img)

        output = []
        for img_array in images:
            output.append(img_array[0])
            chns = img_array[1]
            for chn in chns:
                output.append(np.dstack((chn,chn,chn)))

        return (self.tile_view_16(output), names)

    def draw_pipeline(self, image, chn_edge_phv, title, road):
        '''draw 4 states in pipeline

            Args:
                image (HxWxC)(C:RGB): numpy array image with dimension (height, width, 3)
                chn_edge_phv (HxW): single channel image (8-bit)
                title (string): title
                road (RoadManager): instance of RoadManager

            Returns:
                image (HxWxC)(C:RGB): output image
        '''
        persp_trans = PerspectiveTrans()
        chn_edge_bev = persp_trans.warp(chn_edge_phv)
        conf = road.detect(chn_edge_bev)

        img_bin_phv_raw = np.copy(image)
        img_bin_bev_raw = persp_trans.warp(img_bin_phv_raw)
        img_bin_bev_aux = img_bin_bev_raw

        if conf > 0:
            img_bin_bev_aux = self.draw_trans_box(img_bin_bev_raw, persp_trans)

        img_bin_bev_det = self.draw_lane_scan_windows(img_bin_bev_raw, road)

        if conf > 0:
            img_bin_bev_det = self.draw_lane_line(img_bin_bev_det, road, (0,255,0), 200, 0.2)
            img_bin_bev_det = self.draw_lane_line(img_bin_bev_det, road, (255,0,0), 2)
            img_bin_bev_det = self.draw_selected_pixels(img_bin_bev_det, road, ((0,255,255),(0,0,255)), 0.4)
            img_bin_bev_det = self.draw_lane_area(img_bin_bev_det, road, (0,255,0), 0.2)

        img_bin_phv_aux = img_bin_phv_raw
        if conf > 0:
            img_bin_phv_aux = self.draw_trans_box(img_bin_phv_raw, persp_trans)

        img_bin_phv_det = np.copy(img_bin_phv_aux)
        img_bin_bev_clean = np.zeros_like(img_bin_phv_aux)

        if conf > 0:
            img_bin_bev_clean = self.draw_lane_line(img_bin_bev_clean, road, (0,255,0), 200)
            img_bin_bev_clean = self.draw_lane_line(img_bin_bev_clean, road, (255,0,0), 5)
            img_bin_bev_clean = self.draw_selected_pixels(img_bin_bev_clean, road, ((0,255,255),(0,0,255)))

        img_bin_bev_clean = persp_trans.dewarp(img_bin_bev_clean)
        img_bin_phv_det = cv2.addWeighted(img_bin_phv_det, 0.3, img_bin_bev_clean, 0.7, 0)

        img_bin_phv_final = np.copy(img_bin_phv_raw)
        img_bin_bev_clean = np.zeros_like(img_bin_phv_final)

        if conf > 0:
            img_bin_bev_clean = self.draw_lane_line(img_bin_bev_clean, road, (255,0,0), 5)
            img_bin_bev_clean = self.draw_selected_pixels(img_bin_bev_clean, road, ((0,255,255),(0,0,255)))
            img_bin_bev_clean = self.draw_lane_area(img_bin_bev_clean, road, (0,255,0))

        img_bin_bev_clean = persp_trans.dewarp(img_bin_bev_clean)
        img_bin_phv_final = cv2.addWeighted(img_bin_phv_final, 1, img_bin_bev_clean, 1, 0)

        images = [chn_edge_phv, chn_edge_bev, img_bin_bev_det, img_bin_phv_final]
        return self.blit_in_row(images)


if __name__ == "__main__":
    images_fn = glob.glob("../test_images/*.jpg")
    for img_fn in images_fn:
        print(img_fn)
        img = mpimg.imread(img_fn)
        show_all_channels(img)
