import argparse
import csv
import os
from pathlib import Path

import wisardpkg as wsd

from src.lib.utils import pre_process

parser = argparse.ArgumentParser()
parser.add_argument('--emotion', type=str, default='../data/dataset/emotion.wpkds')
parser.add_argument('--test', type=str, default='../data/omg_ValidationVideos.csv')
parser.add_argument('--out', type=str, default='../data/emotion_output.csv')
parser.add_argument('--videos', type=str, default='../data/videos/')
args = parser.parse_args()


def main(emotion, test, out, videos):
    owd = os.getcwd()

    # Carrega dataset com emotion
    os.chdir(emotion.parent)
    emotion_ds = wsd.DataSet(str(emotion.name))
    os.chdir(owd)

    # Treina Wisard para emotion
    emotion_net = wsd.Wisard(20)
    emotion_net.train(emotion_ds)

    with open(test, 'r', newline='\n') as f:
        with open(out, 'w', newline='\n') as of:
            writer = csv.writer(of, delimiter=',')
            writer.writerow(['video', 'utterance', 'emotion'])
            of.flush()
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                video_id = row[3]
                utterance_id = row[4]
                utterance = videos / video_id / 'video' / utterance_id
                if utterance.is_file():
                    test_ds = wsd.DataSet()
                    test_ds.add(pre_process(utterance))
                    emotion_predict = emotion_net.predict(test_ds)[0]
                    writer.writerow([video_id, utterance_id, emotion_predict])
                    of.flush()


if __name__ == '__main__':
    main(Path(args.emotion), Path(args.test), Path(args.out), Path(args.videos))
