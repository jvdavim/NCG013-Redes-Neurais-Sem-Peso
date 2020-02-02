import os
import sys
from pathlib import Path

import cv2

from lib.yolo.face_detection import get_outputs_names, get_face_boxes

# -------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416
YOLO_CFG = str(Path(os.getcwd()) / Path('lib/yolo/cfg/yolov3-face.cfg'))
YOLO_WEIGHTS = str(Path(os.getcwd()) / Path('lib/yolo/yolo-weights/yolov3-wider_16000.weights'))


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------
def load_yolonet():
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_video(video):
    if video:
        if not video.is_file():
            print('[!] ==> Input video file' + str(video) + 'doesn\'t exist')
            sys.exit(1)
        cap = cv2.VideoCapture(str(video))
    else:
        print('[i] ==> Input file must be a video.')
        sys.exit(1)
    return cap


def crop_face(frame, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))
    # Remove the bounding boxes with low confidence and get face bounds
    faces = get_face_boxes(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    for face in faces:
        xi = max(face[1], 0)
        xf = min(face[1] + face[3], frame.shape[0])
        yi = max(face[0], 0)
        yf = min(face[0] + face[2], frame.shape[1])
        return frame[xi:xf, yi:yf]


def string2json(string, json):
    with open(json, "w") as file:
        file.write(string)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def diff(full_df, partial_df, columns):
    df1 = full_df.loc[:, columns]
    df2 = partial_df.loc[:, columns]
    return full_df.iloc[df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1))].index, :]
