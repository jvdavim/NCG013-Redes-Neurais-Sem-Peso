import argparse

from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./videos',
                    help='path to inputs directory')
parser.add_argument('--output', type=str, default='./outputs',
                    help='path to outputs directory')
args = parser.parse_args()


def detect_faces(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    return


if __name__ == "__main__":
    detect_faces(args.input, args.output)
