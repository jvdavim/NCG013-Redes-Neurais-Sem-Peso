import argparse
from pathlib import Path

import cv2

from src.lib.utils import mkdir, load_network
from src.lib.yolo.face_detection import get_face_frame

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../data/frames',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='../data/dados_proc_1',
                    help='path to frames directory')
args = parser.parse_args()


def main(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    mkdir(out_dir)

    net = load_network()

    for frame_path in in_dir.rglob('*.jpg'):
        in_frame = cv2.imread(str(frame_path))
        out_frame_dir = out_dir / Path(*Path(frame_path).parts[-4:-1])
        mkdir(out_frame_dir)
        out_frame = get_face_frame(in_frame, net)
        in_frame_name = frame_path.name
        try:
            cv2.imwrite(str(out_frame_dir) + '/' + in_frame_name, out_frame)
        except Exception as e:
            print(f'Nao foi possivel recortar a face para o frame {str(out_frame_dir)}/{in_frame_name}: {e}')
            cv2.imwrite(str(out_frame_dir) + '/' + in_frame_name, in_frame)


if __name__ == "__main__":
    main(args.input, args.output)
