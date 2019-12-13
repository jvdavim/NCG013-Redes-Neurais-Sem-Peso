import argparse
import csv
import os
from pathlib import Path

import cv2
import wisardpkg as wsd

from kernel_canvas import apply_kernel_canvas
from lbp import lbp
from lib.utils import load_video, load_network
from lib.yolo.face_detection import get_face_frame
from luminance import get_luminance

parser = argparse.ArgumentParser()
parser.add_argument('--emotion', type=str, default='../data/dataset/emotion.wpkds')
parser.add_argument('--test', type=str, default='../data/omg_ValidationVideos.csv')
parser.add_argument('--out', type=str, default='../data/emotion_output.csv')
parser.add_argument('--videos', type=str, default='../data/videos/')
args = parser.parse_args()


def main(emotion, test, out, videos):
    owd = os.getcwd()

    # Carrega dataset com emotion
    os.chdir(emotion.parent)
    emotion_ds = wsd.DataSet(str(emotion.name))
    os.chdir(owd)

    # Treina Wisard para emotion
    emotion_net = wsd.Wisard(20)
    emotion_net.train(emotion_ds)

    with open(test, 'r', newline='\n') as f:
        with open(out, 'w', newline='\n') as of:
            writer = csv.writer(of, delimiter=',')
            writer.writerow(['video', 'utterance', 'emotion'])
            of.flush()
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                video_id = row[3]
                utterance_id = row[4]
                utterance = videos / video_id / 'video' / utterance_id
                if utterance.is_file():
                    test_ds = get_dataset(utterance)
                    emotion_predict = emotion_net.predict(test_ds)[0]
                    writer.writerow([video_id, utterance_id, emotion_predict])
                    of.flush()


def get_dataset(utterance):
    #  Retorna dataset dado uma utterance (linha/row do csv de teste)
    ds = wsd.DataSet()
    net = load_network()
    cap = load_video(utterance)
    has_frame, frame = cap.read()
    frames = []
    count = 0
    while has_frame:
        try:
            frame = get_face_frame(frame, net)
            frame = cv2.resize(frame, dsize=(91, 124), interpolation=cv2.INTER_CUBIC)
            frame = get_luminance(frame)
            frame = lbp(frame)
            frames += [frame.flatten()]
            print(f'Video: {utterance.parts[3]} \t Utterance: {utterance.name} \t Frame: {count}')
            count += 1
        except Exception as e:
            print(f'Erro ao pre processar frame {count} da utterance: {utterance.name}')
            print(e)
        has_frame, frame = cap.read()
    x = apply_kernel_canvas(frames)
    ds.add(wsd.BinInput(x))
    return ds


if __name__ == '__main__':
    main(Path(args.emotion), Path(args.test), Path(args.out), Path(args.videos))
