from vd.agent import VehicleDetectAgent
from moviepy.editor import VideoFileClip
from vd.viz_util import VisualUtil

import argparse
import os

import logging
logger = logging.getLogger(__name__)

agent = VehicleDetectAgent(0.6, 0)
agent.load()
viz = VisualUtil()

def pipeline_0(img):
    agent.threshold_cls = 0.5
    agent.threshold_heatmap = 0
    regions = agent.action(img)
    img = viz.draw_boxes(img, regions, color=(0, 0, 255), thick=2)
    return img

def pipeline_1(img):
    agent.threshold_cls = 0.95
    agent.threshold_heatmap = 0
    regions = agent.action(img)
    img = viz.draw_boxes(img, regions, color=(0, 0, 255), thick=2)
    return img

pipeline = pipeline_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='sdcp5.py', usage='python %(prog)s [-f] filename [-d]', \
                                     description='Udacity SDC Project 5: Vehicle Detection')
    parser.add_argument('-f', type=str, default='./project_video.mp4', help='input video file name, defult is project_video.mp4')
    args = parser.parse_args()
    v_in_fn = args.f
    fn = os.path.split(v_in_fn)[1]
    v_out_fn = "out_" + fn
    clip = VideoFileClip(v_in_fn)
    processed_clip = clip.fl_image(pipeline)
    processed_clip.write_videofile(v_out_fn, audio=False)
