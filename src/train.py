import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import wisardpkg as wsd

from lib.utils import load_wsd, pre_process, string2json

parser = argparse.ArgumentParser()
parser.add_argument('--videos', type=str, default='../data/videos',
                    help='Diretório com videos organizados conforme na challange')
parser.add_argument('--train', type=str, default='../data/omg_TrainVideos.csv',
                    help='Caminho do arquivo csv para treino')
parser.add_argument('--arousal_net', type=str, default='../tmp/arousal_net.json',
                    help='Caminho do arquivo json da rede treinada para a label arousal')
parser.add_argument('--valence_net', type=str, default='../tmp/valence_net.json',
                    help='Caminho do arquivo json da rede treinada para a label valence')
parser.add_argument('--emotion_net', type=str, default='../tmp/emotion_net.json',
                    help='Caminho do arquivo json da rede treinada para a label emotion')
parser.add_argument('--index', type=str, default='../tmp/index.csv',
                    help='Caminho do arquivo que armazena registros já treinados')
args = parser.parse_args()

videos_dir = Path(args.videos)
train_csv = Path(args.train)
arousal_json = Path(args.arousal_net)
valence_json = Path(args.valence_net)
emotion_json = Path(args.emotion_net)
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
    net = load_wsd(arousal_json, valence_json, emotion_json, 20)
    arousal_net = net[0]
    valence_net = net[1]
    emotion_net = net[2]
    df = full_df[~full_df.isin(index_df).all(axis=1)]
    print(f'[!] ==> Estado recuperado com suceusso')
else:
    df = full_df
    print(f'[!] ==> Iniciando treinamento do zero')
del full_df, args, parser, train_csv
############################################

with open(index_csv, 'a') as index_file:
    index_writer = csv.writer(index_file)
    for index, row in df.iterrows():
        utterance = videos_dir / Path(row[3]) / Path('video') / Path(row[4])
        data = pre_process(utterance)
        arousal_ds, valence_ds, emotion_ds = wsd.DataSet(), wsd.DataSet(), wsd.DataSet()
        arousal_ds.add(data, row[5])
        valence_ds.add(data, row[6])
        emotion_ds.add(data, str(int(row[7])))
        arousal_net.train(arousal_ds)
        valence_net.train(valence_ds)
        emotion_net.train(emotion_ds)
        string2json(arousal_net.json(), arousal_json)
        string2json(valence_net.json(), valence_json)
        string2json(emotion_net.json(), emotion_json)
        index_writer.writerow(row)
        index_file.flush()
