import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import wisardpkg as wsd

from lib.utils import load_wsd, pre_process

# parser = argparse.ArgumentParser()
# parser.add_argument('--videos', type=str, default='../data/videos',
#                     help='Diretório com videos organizados conforme na challange')
# parser.add_argument('--test', type=str, default='../data/omg_TestVideos.csv',
#                     help='Caminho do arquivo csv para teste')
# parser.add_argument('--arousal_net', type=str, default='../data/model/arousal_net.json',
#                     help='Caminho do arquivo json da rede treinada para a label arousal')
# parser.add_argument('--valence_net', type=str, default='../data/model/valence_net.json',
#                     help='Caminho do arquivo json da rede treinada para a label valence')
# parser.add_argument('--emotion_net', type=str, default='../data/model/emotion_net.json',
#                     help='Caminho do arquivo json da rede treinada para a label emotion')
# parser.add_argument('--index', type=str, default='../data/model/index.csv',
#                     help='Caminho do arquivo que armazena registros já treinados')
# args = parser.parse_args()

# videos_dir = Path(args.videos)
# test_csv = Path(args.train)
# arousal_json = Path(args.arousal_net)
# valence_json = Path(args.valence_net)
# emotion_json = Path(args.emotion_net)
# index_csv = Path(args.index)

############################################
# RECUPERA ESTADO ONDE PAROU O TREINAMENTO #
############################################
# if test_csv.is_file():
#     full_df = pd.read_csv(test_csv, sep=',')
# else:
#     print(f'[!] ==> Arquivo de treino nao encontrado')
#     sys.exit(1)
# if index_csv.is_file():
#     index_df = pd.read_csv(index_csv, sep=',',
#                            columns=['link', 'start', 'end', 'video', 'utterance', 'arousal', 'valence',
#                                     'EmotionMaxVote'])
#     net = load_wsd(arousal_json, valence_json, emotion_json, 20)
#     arousal_net = net[0]
#     valence_net = net[1]
#     emotion_net = net[2]
#     df = index_df.merge(full_df, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'right_only']
#     print(f'[!] ==> Iniciando teste com estado recuperado')
# else:
#     df = full_df
#     print(f'[!] ==> Iniciando treinamento do zero')
############################################
#
# with open(index_csv, 'a') as index_file:
#     index_writer = csv.writer(index_file)
#     for row in df.iterrows():
#         utterance = Path('')  # TODO
#         data = pre_process(utterance)
#         arousal_ds, valence_ds, emotion_ds = wsd.DataSet(), wsd.DataSet(), wsd.DataSet()
#         arousal_ds.add(data)
#         valence_ds.add(data)
#         emotion_ds.add(data)
#         arousal_net.train(arousal_ds)
#         valence_net.train(valence_ds)
#         emotion_net.train(emotion_ds)
#         del arousal_ds, valence_ds, emotion_ds, data, utterance
#         index_writer.writerow(row)
