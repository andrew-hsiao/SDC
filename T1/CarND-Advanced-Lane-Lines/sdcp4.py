import os
import argparse
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from alld import cam
from alld import image_filter
from alld import trans
from alld import road_mgr
from alld import viz_util

from alld.cam import CameraAuxiliary
from alld.image_filter import EdgeDetector
from alld.trans import PerspectiveTrans
from alld.road_mgr import LaneLine, RoadManager
from alld.viz_util import VisualUtil

import logging
logger = logging.getLogger(__name__)

cam_aux = CameraAuxiliary(CameraAuxiliary.DEFAULT_CALBRIATION_FILENAME)
edge_dector = EdgeDetector()
persp_trans = PerspectiveTrans()
road = RoadManager((720,1280))
visual = VisualUtil()

def pipeline_diag(img):
    img_clr_phv_cal = cam_aux.calibrate(img)
    chn_edge_phv = edge_dector.detect(img_clr_phv_cal)
    chn_edge_bev = persp_trans.warp(chn_edge_phv)
    road.detect(chn_edge_bev)

    img_clr_bev_raw = persp_trans.warp(img_clr_phv_cal)
    img_clr_bev_aux = visual.draw_trans_box(img_clr_bev_raw, persp_trans)
    #plt.imshow(img_clr_bev_aux)
    #plt.show()

    img_clr_bev_det = visual.draw_lane_scan_windows(img_clr_bev_raw, road)

    img_clr_bev_det = visual.draw_lane_line(img_clr_bev_det, road, (0,255,0), 200, 0.2)
    img_clr_bev_det = visual.draw_lane_line(img_clr_bev_det, road, (255,0,0), 2)
    img_clr_bev_det = visual.draw_selected_pixels(img_clr_bev_det, road, ((0,255,255),(0,0,255)), 0.4)
    img_clr_bev_det = visual.draw_lane_area(img_clr_bev_det, road, (0,255,0), 0.2)
    #plt.imshow(img_clr_bev_det)
    #plt.show()

    img_clr_phv_aux = visual.draw_trans_box(img_clr_phv_cal, persp_trans)
    #plt.imshow(img_clr_phv_aux)
    #plt.show()

    img_clr_phv_det = np.copy(img_clr_phv_aux)
    img_clr_bev_clean = np.zeros_like(img_clr_phv_aux)
    img_clr_bev_clean = visual.draw_lane_line(img_clr_bev_clean, road, (0,255,0), 200)
    img_clr_bev_clean = visual.draw_lane_line(img_clr_bev_clean, road, (255,0,0), 5)
    img_clr_bev_clean = visual.draw_selected_pixels(img_clr_bev_clean, road, ((0,255,255),(0,0,255)))
    img_clr_bev_clean = persp_trans.dewarp(img_clr_bev_clean)
    img_clr_phv_det = cv2.addWeighted(img_clr_phv_det, 0.3, img_clr_bev_clean, 0.7, 0)
    #plt.imshow(img_clr_phv_det)
    #plt.show()

    img_clr_phv_final = np.copy(img_clr_phv_cal)
    img_clr_bev_clean = np.zeros_like(img_clr_phv_final)
    img_clr_bev_clean = visual.draw_lane_line(img_clr_bev_clean, road, (255,0,0), 5)
    img_clr_bev_clean = visual.draw_selected_pixels(img_clr_bev_clean, road, ((0,255,255),(0,0,255)))
    img_clr_bev_clean = visual.draw_lane_area(img_clr_bev_clean, road, (0,255,0))
    img_clr_bev_clean = persp_trans.dewarp(img_clr_bev_clean)
    img_clr_phv_final = cv2.addWeighted(img_clr_phv_final, 1, img_clr_bev_clean, 1, 0)

    if road.is_sraight():
        msg_roc = "Almost straight"
    else:
        msg_roc = "Radius of Curvature = {0:.2f}(m)".format(road.radius_in_meter())

    img_clr_phv_final = visual.draw_text(img_clr_phv_final, msg_roc, (100, 100))
    offset = road.offset_in_meter()

    if offset > 0:
        shift_msg = "left"
    elif offset < 0:
        shift_msg = "right"
    else:
        shift_msg = "no"

    msg_offset = "Vechicle is {0:.2f}cm {1} shift".format(offset*100, shift_msg)
    img_clr_phv_final = visual.draw_text(img_clr_phv_final, msg_offset, (100, 170))

    #plt.imshow(img_clr_phv_final)
    #plt.show()

    img_bin_phv_raw = np.dstack((chn_edge_phv,chn_edge_phv,chn_edge_phv))*255
    #plt.imshow(img_bin_phv_raw)
    #plt.show()

    img_bin_bev_raw = persp_trans.warp(img_bin_phv_raw)
    img_bin_bev_aux = visual.draw_trans_box(img_bin_bev_raw, persp_trans)
    #plt.imshow(img_bin_bev_aux)
    #plt.show()

    img_bin_bev_det = visual.draw_lane_scan_windows(img_bin_bev_raw, road)
    img_bin_bev_det = visual.draw_lane_line(img_bin_bev_det, road, (0,255,0), 200, 0.2)
    img_bin_bev_det = visual.draw_lane_line(img_bin_bev_det, road, (255,0,0), 2)
    img_bin_bev_det = visual.draw_selected_pixels(img_bin_bev_det, road, ((0,255,255),(0,0,255)), 0.4)
    img_bin_bev_det = visual.draw_lane_area(img_bin_bev_det, road, (0,255,0), 0.2)
    #plt.imshow(img_bin_bev_det)
    #plt.show()

    img_bin_phv_aux = visual.draw_trans_box(img_bin_phv_raw, persp_trans)
    #plt.imshow(img_bin_phv_aux)
    #plt.show()

    img_bin_phv_det = np.copy(img_bin_phv_aux)
    img_bin_bev_clean = np.zeros_like(img_bin_phv_aux)
    img_bin_bev_clean = visual.draw_lane_line(img_bin_bev_clean, road, (0,255,0), 200)
    img_bin_bev_clean = visual.draw_lane_line(img_bin_bev_clean, road, (255,0,0), 5)
    img_bin_bev_clean = visual.draw_selected_pixels(img_bin_bev_clean, road, ((0,255,255),(0,0,255)))
    img_bin_bev_clean = persp_trans.dewarp(img_bin_bev_clean)
    img_bin_phv_det = cv2.addWeighted(img_bin_phv_det, 0.3, img_bin_bev_clean, 0.7, 0)
    #plt.imshow(img_bin_phv_det)
    #plt.show()

    img_bin_phv_final = np.copy(img_bin_phv_raw)
    img_bin_bev_clean = np.zeros_like(img_bin_phv_final)
    img_bin_bev_clean = visual.draw_lane_line(img_bin_bev_clean, road, (255,0,0), 5)
    img_bin_bev_clean = visual.draw_selected_pixels(img_bin_bev_clean, road, ((0,255,255),(0,0,255)))
    img_bin_bev_clean = visual.draw_lane_area(img_bin_bev_clean, road, (0,255,0))
    img_bin_bev_clean = persp_trans.dewarp(img_bin_bev_clean)
    img_bin_phv_final = cv2.addWeighted(img_bin_phv_final, 1, img_bin_bev_clean, 1, 0)

    msg = "Confidence: {0:.2f}".format(road.confidence())
    img_bin_phv_final = visual.draw_text(img_bin_phv_final, msg, (100, 100), 2, (255,0,255), 5)

    msg = "Frame: {0:9d}".format(road.detect_counter)
    img_bin_phv_final = visual.draw_text(img_bin_phv_final, msg, (100, 170), 2, (255,0,255), 5)

    #plt.imshow(img_bin_phv_final)
    #plt.show()

    all_images = [img_clr_phv_final,
                    img_bin_phv_final,
                    img_clr_phv_aux,
                    img_clr_bev_aux,
                    img_clr_phv_det,
                    img_clr_bev_det,
                    img_bin_phv_aux,
                    img_bin_bev_aux,
                    img_bin_phv_det,
                    img_bin_bev_det]

    image_diag = visual.tile_view_10(all_images)

    return image_diag

def pipeline(img):
    img_clr_phv_cal = cam_aux.calibrate(img)
    chn_edge_phv = edge_dector.detect(img_clr_phv_cal)
    chn_edge_bev = persp_trans.warp(chn_edge_phv)
    conf = road.detect(chn_edge_bev)

    img_clr_phv_final = np.copy(img_clr_phv_cal)
    img_clr_bev_clean = np.zeros_like(img_clr_phv_final)
    img_clr_bev_clean = visual.draw_lane_line(img_clr_bev_clean, road, (255,0,0), 5)
    img_clr_bev_clean = visual.draw_selected_pixels(img_clr_bev_clean, road, ((0,255,255),(0,0,255)))
    img_clr_bev_clean = visual.draw_lane_area(img_clr_bev_clean, road, (0,255,0))
    img_clr_bev_clean = persp_trans.dewarp(img_clr_bev_clean)
    img_clr_phv_final = cv2.addWeighted(img_clr_phv_final, 1, img_clr_bev_clean, 1, 0)

    if road.is_sraight():
        msg_roc = "Almost straight"
    else:
        msg_roc = "Radius of Curvature = {0:.2f}(m)".format(road.radius_in_meter())

    img_clr_phv_final = visual.draw_text(img_clr_phv_final, msg_roc, (100, 100))
    offset = road.offset_in_meter()

    if offset > 0:
        shift_msg = "left"
    elif offset < 0:
        shift_msg = "right"
    else:
        shift_msg = "no"

    msg_offset = "Vechicle is {0:.2f}cm {1} shift".format(offset*100, shift_msg)
    img_clr_phv_final = visual.draw_text(img_clr_phv_final, msg_offset, (100, 170))
    return img_clr_phv_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='sdcp4.py', usage='python %(prog)s [-f] filename [-d]', \
                                     description='Udacity SDC Project 4: Advanced Lane Finding')
    parser.add_argument('-d', action='store_true', help='use diagnostic pipeline')
    parser.add_argument('-f', type=str, default='./project_video.mp4', help='input video file name, defult is project_video.mp4')
    args = parser.parse_args()
    v_in_fn = args.f
    fn = os.path.split(v_in_fn)[1]
    if args.d:
        v_out_fn = "out_diag_" + fn
        clip = VideoFileClip(v_in_fn)
        processed_clip = clip.fl_image(pipeline_diag)
    else:
        v_out_fn = "out_" + fn
        clip = VideoFileClip(v_in_fn)
        processed_clip = clip.fl_image(pipeline)
    processed_clip.write_videofile(v_out_fn, audio=False)
