import cv2
import sys
import argparse
import numpy as np

from wisardpkg import KernelCanvas
from pathlib import Path
from utils import mkdir


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./videos',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='./outputs',
                    help='path to outputs directory')
parser.add_argument('--csv', type=str, default='./data/omg_TrainVideos.csv',
                    help='path to outputs directory')
args = parser.parse_args()


def applyKernelCanvas(frame):
    numberOfKernels = 10
    dimension = 2
    kc = KernelCanvas(
        dimension,               # required
        numberOfKernels,         # required
        bitsBykernel=3,        # optional
        activationDegree=0.07,  # optional
        useDirection=False     # optional
    )
    return kc.transform(frame.astype(int))


def main(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    mkdir(out_dir)

    with open(args.csv, 'r') as in_csv:
        for frame_path in in_dir.rglob('*.jpg'):
            frame = cv2.imread(str(frame_path), 0)
            out = applyKernelCanvas(frame)
            print("hello")


if __name__ == '__main__':
    main(args.input, args.output)
