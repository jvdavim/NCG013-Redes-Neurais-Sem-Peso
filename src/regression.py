import argparse
import csv
import os
from pathlib import Path

import cv2
import wisardpkg as wsd

from kernel_canvas import apply_kernel_canvas
from lbp import lbp
from lib.utils import load_yolonet, load_video
from lib.yolo.face_detection import get_face_frame
from luminance import get_luminance

parser = argparse.ArgumentParser()
parser.add_argument('--arousal', type=str, default='../data/dataset/arousal.wpkds')
parser.add_argument('--valence', type=str, default='../data/dataset/valence.wpkds')
parser.add_argument('--test', type=str, default='../data/omg_ValidationVideos.csv')
parser.add_argument('--out', type=str, default='../data/model_output.csv')
parser.add_argument('--videos', type=str, default='../data/videos/')
args = parser.parse_args()


def main(arousal, valence, test, out, videos):
    owd = os.getcwd()

    # Carrega dataset com arousal
    os.chdir(arousal.parent)
    arousal_ds = wsd.DataSet(str(arousal.name))
    os.chdir(owd)

    # Carrega dataset com valence
    os.chdir(str(valence.parent))
    valence_ds = wsd.DataSet(str(valence.name))
    os.chdir(owd)

    # Treina RegressionWisard para aruousal
    arousal_net = wsd.RegressionWisard(20)
    arousal_net.train(arousal_ds)

    # Treina RegressionWisard para valence
    valence_net = wsd.RegressionWisard(20)
    valence_net.train(valence_ds)

    os.chdir(test.parent)
    with open(test.name, 'r', newline='\n') as f:
        with open(out.name, 'w', newline='\n') as of:
            writer = csv.writer(of, delimiter=',')
            writer.writerow(['video', 'utterance', 'arousal', 'valence'])
            of.flush()
            reader = csv.reader(f)
            next(reader)
            os.chdir(owd)
            for row in reader:
                video_id = row[3]
                utterance_id = row[4]
                utterance = videos / video_id / 'video' / utterance_id
                if utterance.is_file():
                    test_ds = get_dataset(utterance)
                    arousal_predict = arousal_net.predict(test_ds)[0]
                    valence_predict = valence_net.predict(test_ds)[0]
                    writer.writerow([video_id, utterance_id, arousal_predict, valence_predict])
                    of.flush()


def get_dataset(utterance):
    #  Retorna dataset dado uma utterance (linha/row do csv de teste)
    ds = wsd.DataSet()
    net = load_yolonet()
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
            print(f'[!] ==> Video: {utterance.parts[3]} \t Utterance: {utterance.name} \t Frame: {count}')
            count += 1
        except Exception as e:
            print(f'[!] ==> Erro ao pre processar frame {count} da utterance: {utterance.name}')
            print(e)
        has_frame, frame = cap.read()
    x = apply_kernel_canvas(frames)
    ds.add(wsd.BinInput(x))
    return ds


if __name__ == '__main__':
    main(Path(args.arousal), Path(args.valence), Path(args.test), Path(args.out), Path(args.videos))
