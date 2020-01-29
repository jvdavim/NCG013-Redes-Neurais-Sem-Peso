import os
import sys
from pathlib import Path

import cv2
import numpy as np
import wisardpkg as wsd
# -------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
from skimage import feature

from lib.yolo.face_detection import get_outputs_names, get_face_boxes

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416
YOLO_CFG = str(Path(os.getcwd()) / Path('lib/yolo/cfg/yolov3-face.cfg'))
YOLO_WEIGHTS = str(Path(os.getcwd()) / Path('lib/yolo/yolo-weights/yolov3-wider_16000.weights'))


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

def load_wsd(arousal_json, valence_json, emotion_json, param):
    net = list()
    if arousal_json.is_file():
        with open(arousal_json) as file:
            arousal_net = wsd.RegressionWisard(file.read())
    else:
        arousal_net = wsd.RegressionWisard(param)
    net.append(arousal_net)
    if valence_json.is_file():
        with open(valence_json) as file:
            valence_net = wsd.RegressionWisard(file.read())
    else:
        valence_net = wsd.RegressionWisard(param)
    net.append(valence_net)
    if emotion_json.is_file():
        with open(emotion_json) as file:
            emotion_net = wsd.Wisard(file.read())
    else:
        emotion_net = wsd.Wisard(param)
    net.append(emotion_net)
    return tuple(net)


def crop(utterance):
    net = load_yolonet()
    cap = load_video(utterance)
    has_frame, frame = cap.read()
    count = 0
    while has_frame:
        try:
            frame = crop_face(frame, net)
            return frame
        except Exception as e:
            print(f'[!] ==> Erro ao pre processar frame {count} da utterance: {utterance.name}')
            print(f'[E] ==> {e}')
            return frame
        has_frame, frame = cap.read()


def pre_process(utterance):
    net = load_yolonet()
    cap = load_video(utterance)
    has_frame, frame = cap.read()
    frames = []
    count = 0
    while has_frame:
        try:
            frame = crop_face(frame, net)
            frame = cv2.resize(frame, dsize=(91, 124), interpolation=cv2.INTER_CUBIC)
            frame = get_luminance(frame)
            frame = lbp(frame)
            frames += [frame.flatten()]
            print(f'[!] ==> Video: {utterance.parts[3]} \t Utterance: {utterance.name} \t Frame: {count}')
            count += 1
        except Exception as e:
            print(f'[!] ==> Erro ao pre processar frame {count} da utterance: {utterance.name}')
            print(f'[E] ==> {e}')
        has_frame, frame = cap.read()
    x = apply_kernel_canvas(frames)
    return wsd.BinInput(x)


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


def get_luminance(frame):
    # 0.2126*R + 0.7152*G + 0.0722*B
    w = np.array([[[0.0722, 0.7152, 0.2126]]])
    luminance = cv2.convertScaleAbs(np.sum(frame * w, axis=2))
    return luminance


def lbp(frame):
    frame = feature.local_binary_pattern(frame, 24, 8, method='uniform')
    return frame


def apply_kernel_canvas(frames):
    number_of_kernels = 100
    dimension = len(frames[0])

    kc = wsd.KernelCanvas(
        dimension,  # required
        number_of_kernels,  # required
        bitsBykernel=30,  # optional
        activationDegree=0.07,  # optional
        useDirection=False  # optional
    )
    return kc.transform(frames)


def string2json(string, json):
    with open(json, "w") as file:
        file.write(string)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def diff(full_df, partial_df, columns):
    df1 = full_df.loc[:, columns]
    df2 = partial_df.loc[:, columns]
    return full_df.iloc[df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1))].index, :]
