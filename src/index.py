import argparse
from pathlib import Path

import pandas as pd

from lib.utils import load_ds, get_df_from_cache

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='../data/videos')
parser.add_argument('--input_csv', type=str, default='../data/omg_TestVideos_WithLabels.csv')
parser.add_argument('--cache', type=str, default='../cache/cache.csv')
parser.add_argument('--arousal_ds', type=str, default='../data/dataset/arousal.wpkds')
parser.add_argument('--valence_ds', type=str, default='../data/dataset/valence.wpkds')
parser.add_argument('--emotion_ds', type=str, default='../data/dataset/emotion.wpkds')
args = parser.parse_args()


def main(input_dir, input_csv, arousal_ds, valence_ds, emotion_ds, cache):
    if cache.is_file():
        df = get_df_from_cache(input_csv, cache)
    else:
        df = pd.read_csv(input_csv)
    ards = load_ds(arousal_ds)
    vads = load_ds(valence_ds)
    emds = load_ds(emotion_ds)
    df = pd.read_csv(input_csv)


if __name__ == '__main__':
    main(Path(args.input_dir), Path(args.input_csv), Path(args.arousal_ds), Path(args.valence_ds),
         Path(args.emotion_ds), Path(args.cache))
