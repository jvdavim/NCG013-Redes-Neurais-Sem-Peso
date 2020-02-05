import argparse
import csv
import sys
from pathlib import Path

import cv2
import pandas as pd

from lib.utils import mkdir, load_yolonet, load_video, crop_face, diff

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', type=str, default='../data/omg_TrainVideos.csv',
                    help='Caminho do arquivo.csv com os dados das utterances')
parser.add_argument('--videos', type=str, default='../data/videos',
                    help='Diretório com videos organizados conforme na challange')
parser.add_argument('--output', type=str, default='../output/videos',
                    help='Diretório onde serão salvas as imagens.jpg com as faces recortadas')
parser.add_argument('--index', type=str, default='../tmp/index.csv',
                    help='Caminho do arquivo que armazena registros já processados')
args = parser.parse_args()

metadata = Path(args.metadata)
videos_dir = Path(args.videos)
output_dir = Path(args.output)
index = Path(args.index)

############################################
#        RECUPERA ESTADO ONDE PAROU        #
############################################
if metadata.is_file():
    full_df = pd.read_csv(metadata, sep=',')
else:
    print(f'[!] ==> Arquivo de treino nao encontrado')
    sys.exit(1)
if index.is_file():
    index_df = pd.read_csv(index, sep=',', names=['video', 'utterance'])
    df = diff(full_df, index_df, ['video', 'utterance'])
    print(f'[!] ==> Estado recuperado com suceusso')
else:
    df = full_df
    print(f'[!] ==> Iniciando processamento do zero')
del full_df, args, parser, metadata
############################################

with open(index, 'a') as index_file:
    index_writer = csv.writer(index_file)
    for i, row in df.iterrows():
        utterance = videos_dir / Path(row[3]) / Path('video') / Path(row[4])
        out_dir = output_dir / Path(row[3]) / Path('video') / Path(row[4]).with_suffix('')
        mkdir(out_dir)
        net = load_yolonet()
        cap = load_video(utterance)
        has_frame, frame = cap.read()
        count = 0
        while has_frame:
            try:
                frame = crop_face(frame, net)
                cv2.imwrite(str(out_dir / Path(f'frame{count}.jpg')), frame)
                print(f'[!] ==> Video: {utterance.parts[3]} \t Utterance: {utterance.name} \t Frame: {count}')
                count += 1
            except Exception as e:
                print(f'[!] ==> Erro ao pre processar frame {count} da utterance: {utterance.name}')
                print(f'[E] ==> {e}')
            has_frame, frame = cap.read()
        index_writer.writerow(row[3:5])
        index_file.flush()
