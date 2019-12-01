import cv2
import sys
import csv
import argparse
import numpy as np

import wisardpkg as wsd
from pathlib import Path
from utils import mkdir


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./dados_proc_3',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='./teste',
                    help='path to outputs directory')
parser.add_argument('--csv', type=str, default='./data/omg_TrainVideos.csv',
                    help='path to outputs directory')
args = parser.parse_args()


def applyKernelCanvas(frame):
    numberOfKernels = 1000
    dimension = len(frame[0])

    kc = wsd.KernelCanvas(
        dimension,               # required
        numberOfKernels,         # required
        bitsBykernel=3,        # optional
        activationDegree=0.07,  # optional
        useDirection=False     # optional
    )
    return kc.transform(frame.astype(int))


def get_utt_dir(in_dir, row):
    return in_dir / row[3] / 'video' / Path(row[4]).with_suffix('')


def main(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    mkdir(out_dir)

    arousal_ds = wsd.DataSet()
    valence_ds = wsd.DataSet()
    emotion_ds = wsd.DataSet()

    with open(args.csv, 'r', newline='\n') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # CAPTURAR DIRETORIO DOS FRAMES DA UTTERANCE ATUAL
            utt_dir = get_utt_dir(in_dir, row)
            # CAPTURAR AROUSAL, VALENCE e EMOTIONMAXVOTE PARA UTTERANCE ATUAL
            arousal = row[5]
            valence = row[6]
            emotion = row[7]
            # PARA CADA FRAME, ADICIONAR UM BININPUT NO DATASET COM A LABEL E O KERNEL CANVAS
            if utt_dir.is_dir():
                for frame_path in utt_dir.rglob('*.jpg'):
                    frame = cv2.imread(str(frame_path), 0)
                    x = applyKernelCanvas(frame)
                    arousal_ds.add(wsd.BinInput(x), arousal)
                    valence_ds.add(wsd.BinInput(x), valence)
                    emotion_ds.add(wsd.BinInput(x), emotion)
    arousal_ds.save(str(out_dir / 'arousal_ds.wpkds'))
    valence_ds.save(str(out_dir / 'valence_ds.wpkds'))
    emotion_ds.save(str(out_dir / 'emotion_ds.wpkds'))


if __name__ == '__main__':
    main(args.input, args.output)
