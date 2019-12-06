import sys
from pathlib import Path

import cv2

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416
YOLO_CFG = './yolo/cfg/yolov3-face.cfg'
YOLO_WEIGHTS = './yolo/yolo-weights/yolov3-wider_16000.weights'


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------


def load_network():
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_video(video):
    if video:
        if not video.is_file():
            print('[!] ==> Input video file' + video + 'doesn\'t exist')
            sys.exit(1)
        cap = cv2.VideoCapture(str(video))
    else:
        print('[i] ==> Input file must be a video.')
    return cap


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
