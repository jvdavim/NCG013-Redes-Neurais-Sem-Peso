import argparse
from pathlib import Path

import cv2
import numpy as np

from lib.utils import mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../data/dados_proc_1',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='../data/dados_proc_1_resized',
                    help='path to frames directory')
args = parser.parse_args()


def get_median_imgsize(in_dir):
    fx = []
    fy = []
    for frame_path in in_dir.rglob('*.jpg'):
        frame = cv2.imread(str(frame_path))
        fx += [len(frame[0])]
        fy += [len(frame)]
    return np.median(fx).astype(int), np.median(fy).astype(int)


def main(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    mkdir(out_dir)

    median_imgsize = (get_median_imgsize(in_dir))

    for frame_path in in_dir.rglob('*.jpg'):
        frame = cv2.imread(str(frame_path))
        frame = cv2.resize(frame, dsize=median_imgsize, interpolation=cv2.INTER_CUBIC)
        frame_dir = out_dir / Path(*Path(frame_path).parts[-4:-1])
        mkdir(frame_dir)
        out_path = str(frame_dir) + '/' + str(frame_path.name)
        cv2.imwrite(out_path, frame)


if __name__ == '__main__':
    main(args.input, args.output)
