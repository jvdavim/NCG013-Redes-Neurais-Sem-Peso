import argparse
import csv
from pathlib import Path

import cv2
import wisardpkg as wsd

from src.lib.utils import mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../data/dados_proc_3',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='./',
                    help='path to frames directory')
parser.add_argument('--csv', type=str, default='../data/omg_TrainVideos.csv',
                    help='path to frames directory')
args = parser.parse_args()


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


def get_utt_dir(in_dir, row):
    return in_dir / row[3] / 'video' / Path(row[4]).with_suffix('')


def main(in_dir, out_dir, csv_path):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)
    csv_path = Path(csv_path)

    mkdir(out_dir)

    arousal_ds = wsd.DataSet()
    valence_ds = wsd.DataSet()
    emotion_ds = wsd.DataSet()

    with open(csv_path, 'r', newline='\n') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # CAPTURAR DIRETORIO DOS FRAMES DA UTTERANCE ATUAL
            utt_dir = get_utt_dir(in_dir, row)
            # CAPTURAR AROUSAL, VALENCE e EMOTIONMAXVOTE PARA UTTERANCE ATUAL
            arousal = float(row[5])
            valence = float(row[6])
            emotion = str(row[7])

            if utt_dir.is_dir():
                # CAPTURAR COLECAO DE FRAMES
                frames = []
                for frame_path in utt_dir.rglob('*.jpg'):
                    frame = cv2.imread(str(frame_path), 0)
                    frames += [frame.flatten().tolist()]
                # CRIA REGISTRO PARA UTTERANCE NO DATASET
                x = apply_kernel_canvas(frames)
                arousal_ds.add(wsd.BinInput(x), arousal)
                valence_ds.add(wsd.BinInput(x), valence)
                emotion_ds.add(wsd.BinInput(x), emotion)
    arousal_ds.save(str(out_dir / 'arousal_ds'))
    valence_ds.save(str(out_dir / 'valence_ds'))
    emotion_ds.save(str(out_dir / 'emotion_ds'))


if __name__ == '__main__':
    main(args.input, args.output, args.csv)
