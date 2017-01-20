import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import scipy
import cv2

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
prev_image_array = None
dense_net = None

#input: RGB image, output: BGRG image
def preprocess_insert_gray_channel(img):
    #plt.imshow(final)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r,g,b = cv2.split(img)
    bgrg_img = cv2.merge((b,g,r,gray))
    return bgrg_img

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    #print(image_array.shape)
    camera_frame = image_array[None, :, :, :][0]
    #camera_frame is (160x320x3)
    resized_frame = cv2.resize(camera_frame, (80, 80))
    bgrg_img = preprocess_insert_gray_channel(resized_frame)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = dense_net.predict(np.array([bgrg_img]))[-1,0]
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.8
    #print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    print("steering_angle:{0}, throttle:{1}".format(steering_angle, throttle))
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        dense_net = model_from_json(jfile.read())
        print(dense_net.summary())

    dense_net.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    dense_net.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
