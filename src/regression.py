import argparse
import csv
from pathlib import Path

import cv2
import wisardpkg as wsd

from src.kernel_canvas import apply_kernel_canvas
from src.lbp import lbp
from src.lib.utils import load_video, load_network
from src.lib.yolo.face_detection import get_face_frame
from src.luminance import get_luminance

parser = argparse.ArgumentParser()
parser.add_argument('--arousal_ds', type=str, default='arousal_ds.wpkds')
parser.add_argument('--valence_ds', type=str, default='valence_ds.wpkds')
parser.add_argument('--test_csv', type=str, default='../data/omg_TestVideos_WithLabels.csv')
parser.add_argument('--out_csv', type=str, default='../data/omg_arousal_predictions.csv')
parser.add_argument('--videos_dir', type=str, default='../data/videos/')
args = parser.parse_args()


def main(arousal_path, valence_path, test_csv, out_csv, videos_dir):
    test_csv = Path(test_csv)
    out_csv = Path(out_csv)
    videos_dir = Path(videos_dir)

    arousal_train_ds = wsd.DataSet(str(arousal_path))
    valence_train_ds = wsd.DataSet(str(valence_path))

    arousal_net = wsd.RegressionWisard(20)
    arousal_net.train(arousal_train_ds)

    valence_net = wsd.RegressionWisard(20)
    valence_net.train(valence_train_ds)

    with open(test_csv, 'r', newline='\n') as f:
        with open(out_csv, 'w', newline=',\n') as of:
            writer = csv.writer(of, delimiter=',')
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                test_ds = get_dataset(row, videos_dir)
                arousal_predict = arousal_net.predict(test_ds)
                valence_predict = valence_net.predict(test_ds)
                video = row[3]
                utterance = row[4]
                writer.writerow([video, utterance, arousal_predict, valence_predict])


def get_dataset(row, videos_dir):
    #  Retorna dataset dado uma utterance (linha/row do csv de teste)
    ds = wsd.DataSet()
    net = load_network()
    cap = load_video(str(videos_dir / row[3] / 'video' / row[4]))
    has_frame, frame = cap.read()
    frames = []
    while has_frame:
        frame = get_face_frame(frame, net)
        frame = cv2.resize(frame, dsize=(91, 124), interpolation=cv2.INTER_CUBIC)
        frame = get_luminance(frame)
        frame = lbp(frame)
        frames += [frame.flatten()]
    x = apply_kernel_canvas(frames)
    ds.add(wsd.BinInput(x))
    return ds


if __name__ == '__main__':
    main(args.arousal_ds, args.valence_ds, args.test_csv, args.out_csv, args.videos_dir)
