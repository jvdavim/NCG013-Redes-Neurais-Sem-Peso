import os
import cv2
import sys
import argparse

from pathlib import Path
from utils import mkdir, load_network
from yolo.face_detection import get_face_frame


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./videos',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='./outputs',
                    help='path to outputs directory')
args = parser.parse_args()


def main(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    mkdir(out_dir)

    net = load_network()

    for in_frame_path in in_dir.rglob('*.jpg'):
        in_frame = cv2.imread(str(in_frame_path))
        out_frame_dir = out_dir / Path(*Path(os.path.splitext(in_frame_path)[0]).parts[-4:-1])
        mkdir(out_frame_dir)
        out_frame = get_face_frame(in_frame, net)
        in_frame_name = in_frame_path.name
        cv2.imwrite(str(out_frame_dir) + '/' + in_frame_name, out_frame)


if __name__ == "__main__":
    main(args.input, args.output)
