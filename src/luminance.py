import argparse
from pathlib import Path

import cv2
import numpy as np

from src.lib.utils import mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../data/dados_proc_1',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='../data/dados_proc_2',
                    help='path to frames directory')
args = parser.parse_args()


def get_luminance(frame):
    # 0.2126*R + 0.7152*G + 0.0722*B
    w = np.array([[[0.0722, 0.7152, 0.2126]]])
    luminance = cv2.convertScaleAbs(np.sum(frame * w, axis=2))
    return luminance


def main(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    mkdir(out_dir)

    for frame_path in in_dir.rglob('*.jpg'):
        frame = cv2.imread(str(frame_path))
        frame = get_luminance(frame)
        frame_dir = out_dir / Path(*Path(frame_path).parts[-4:-1])
        mkdir(frame_dir)
        out_path = str(frame_dir) + '/' + str(frame_path.name)
        cv2.imwrite(out_path, frame)


if __name__ == '__main__':
    main(args.input, args.output)
