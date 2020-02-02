import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import wisardpkg as wsd

from transformers import LocalBinaryPatternPipeline

# -------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
# Output DataSet
EMOTION_DS = str(Path('./emotion'))
AROUSAL_DS = str(Path('./arousal'))
VALENCE_DS = str(Path('./valence'))

# Kernel Canvas
NUMBER_OF_KERNELS = 100

# Local Binary Pattern
P = 24
R = 8
METHOD = 'uniform'

# -------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../output/videos',
                    help='Diretório com as imagens das faces de cada utterance')
parser.add_argument('--metadata', type=str, default='../data/omg_TrainVideos.csv',
                    help='Caminho do arquivo csv de metadados')
args = parser.parse_args()

data = Path(args.data)
metadata = Path(args.metadata)

# -------------------------------------------------------------------
# Script
# ------------------------------------------------------------------
df = pd.read_csv(metadata, sep=',')

e_ds = wsd.DataSet()
a_ds = wsd.DataSet()
v_ds = wsd.DataSet()

for i, row in df.iterrows():
    img_dir = Path(data) / Path(row[3]) / Path('video') / Path(row[4]).with_suffix('')
    img_paths = glob.glob(str(img_dir / Path('*.jpg')))
    img_set = np.array([], np.uint8)
    for img_path in img_paths:
        if Path(img_path).stat().st_size:
            img = cv2.imread(img_path)
            pipeline = LocalBinaryPatternPipeline(img, P, R, METHOD)
            img = pipeline.transform()  # executa o pre processamento
            img_set = np.concatenate((img_set, img), axis=None)  # concatena todas as imagens em uma so
    if len(img_set) > 0:
        kc = wsd.KernelCanvas(
            len(img_set),  # required
            NUMBER_OF_KERNELS,  # required
            bitsBykernel=30,  # optional
            activationDegree=0.07,  # optional
            useDirection=False  # optional
        )
        x = kc.transform([img_set])
        e_ds.add(wsd.BinInput(x), str(int(row[7])))
        a_ds.add(wsd.BinInput(x), row[5])
        v_ds.add(wsd.BinInput(x), row[6])

e_ds.save(EMOTION_DS)
a_ds.save(AROUSAL_DS)
v_ds.save(VALENCE_DS)
