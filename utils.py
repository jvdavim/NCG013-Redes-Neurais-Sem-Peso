import cv2
import sys
import os

from pathlib import Path

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------


def load_network(model_cfg, model_weights):
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_video(video):
    if video:
        if not os.path.isfile(video):
            print(f'[!] ==> Input video file {video} doesn\'t exist')
            sys.exit(1)
        cap = cv2.VideoCapture(str(video))
    else:
        print(f'[i] ==> Input file must be a video.')
    return cap


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
