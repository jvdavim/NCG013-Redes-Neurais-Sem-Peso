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
        frame_dir = out_dir / Path(*Path(os.path.splitext(utt_path)[0]).parts[-3:])
        mkdir(frame_dir)
        has_frame, frame = cap.read()
        count = 0
        while has_frame:
            cv2.imwrite(f'{str(frame_dir)}/frame_{count}.jpg', frame)
            has_frame, frame = cap.read()
            count += 1


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    extract_frames('./data/videos/', './outputs')
