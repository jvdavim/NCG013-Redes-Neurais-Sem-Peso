import os
import cv2

from pathlib import Path
from utils import load_video


def extract_frames(in_dir, out_dir):
    out_dir = Path(out_dir)
    in_dir = Path(in_dir)

    # Cria diretorio de output
    mkdir(out_dir)

    for utt_path in in_dir.rglob('*.mp4'):
        cap = load_video(utt_path)
        has_frame, frame = cap.read()
        count = 0
        while has_frame:
            frame_dir = out_dir / Path(*Path(os.path.splitext(utt_path)[0]).parts[-3:])
            mkdir(frame_dir)
            frame_path = get_frame_path(out_dir, utt_path, count)
            cv2.imwrite(frame_path, frame)
            has_frame, frame = cap.read()
            count += 1


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_frame_path(out_dir, utt_path, count):
    parts = list(utt_path.parts[-3:])
    filename = f'frame_{count}'
    parts[2] = filename
    return out_dir / Path(*parts)


if __name__ == "__main__":
    extract_frames('./data/videos/', './outputs')
