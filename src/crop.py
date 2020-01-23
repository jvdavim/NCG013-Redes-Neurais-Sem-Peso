import argparse
import csv
import sys
from pathlib import Path

import cv2
import pandas as pd

from src.lib.utils import mkdir, load_yolonet, load_video, crop_face

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/videos',
                    help='Diretório com videos organizados conforme na challange')
parser.add_argument('--output', type=str, default='../output/videos',
                    help='Diretório com frames com as faces recortadas')
parser.add_argument('--train', type=str, default='../data/omg_TrainVideos.csv',
                    help='Caminho do arquivo csv para treino')
parser.add_argument('--index', type=str, default='../tmp/crop_index.csv',
                    help='Caminho do arquivo que armazena registros já treinados')
args = parser.parse_args()

videos_dir = Path(args.data)
output = Path(args.output)
train_csv = Path(args.train)
index_csv = Path(args.index)

############################################
# RECUPERA ESTADO ONDE PAROU O TREINAMENTO #
############################################
if train_csv.is_file():
    full_df = pd.read_csv(train_csv, sep=',')
else:
    print(f'[!] ==> Arquivo de treino nao encontrado')
    sys.exit(1)
if index_csv.is_file():
    index_df = pd.read_csv(index_csv, sep=',',
                           names=['link', 'start', 'end', 'video', 'utterance', 'arousal', 'valence',
                                  'EmotionMaxVote'])
    df = full_df[~full_df.isin(index_df).all(axis=1)]
    print(f'[!] ==> Estado recuperado com suceusso')
else:
    df = full_df
    print(f'[!] ==> Iniciando processamento do zero')
del full_df, args, parser, train_csv
############################################

with open(index_csv, 'a') as index_file:
    index_writer = csv.writer(index_file)
    for index, row in df.iterrows():
        utterance = videos_dir / Path(row[3]) / Path('video') / Path(row[4])
        out_dir = output / Path(row[3]) / Path('video') / Path(row[4]).with_suffix('')
        mkdir(out_dir)
        net = load_yolonet()
        cap = load_video(utterance)
        has_frame, frame = cap.read()
        count = 0
        while has_frame:
            try:
                frame = crop_face(frame, net)
                cv2.imwrite(str(out_dir / Path(f'frame{count}.png')), frame)
                print(f'[!] ==> Video: {utterance.parts[3]} \t Utterance: {utterance.name} \t Frame: {count}')
                count += 1
            except Exception as e:
                print(f'[!] ==> Erro ao pre processar frame {count} da utterance: {utterance.name}')
                print(f'[E] ==> {e}')
            has_frame, frame = cap.read()
        index_writer.writerow(row)
        index_file.flush()
