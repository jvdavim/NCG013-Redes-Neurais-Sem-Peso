import argparse
import sys
from pathlib import Path

import pandas as pd
import wisardpkg as wsd

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../data/videos')
parser.add_argument('--arousal_net', type=str, default='../data/model/arousal_net.json')
parser.add_argument('--valence_net', type=str, default='../data/model/valence_net.json')
parser.add_argument('--emotion_net', type=str, default='../data/model/emotion_net.json')
parser.add_argument('--index', type=str, default='../data/model/index.csv')  # Armazena valores que ja foram treinados
args = parser.parse_args()


def main(in_dir, arousal_json, valence_json, emotion_json, index_csv):
    ############################################
    # RECUPERA ESTADO ONDE PAROU O TREINAMENTO #
    ############################################
    if (in_dir / Path('omg_TrainVideos.csv')).is_file():
        full_df = pd.read_csv(in_dir / Path('omg_TrainVideos.csv'), sep=',')
    else:
        print(f'Arquivo de treino nao encontrado')
        sys.exit(1)
    if index_csv.is_file():
        index_df = pd.read_csv(index_csv, sep=',',
                               columns=['link', 'start', 'end', 'video', 'utterance', 'arousal', 'valence',
                                        'EmotionMaxVote'])
        net = load_net(arousal_json, valence_json, emotion_json, 20)
        arousal_net = net[0]
        valence_net = net[1]
        emotion_net = net[2]
        df = index_df.merge(full_df, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'right_only']
        print(f'Iniciando treinamento com estado recuperado')
    else:
        df = full_df
        print(f'Iniciando treinamento do zero')
    ############################################
    # with open(index_csv, 'a') as index_file:


def load_net(arousal_json, valence_json, emotion_json, param):
    net = list()
    if arousal_json.is_file():
        arousal_net = wsd.RegressionWisard(arousal_json)
    else:
        arousal_net = wsd.RegressionWisard(param)
    net.append(arousal_net)
    if valence_json.is_file():
        valence_net = wsd.RegressionWisard(valence_json)
    else:
        valence_net = wsd.RegressionWisard(param)
    net.append(valence_net)
    if emotion_json.is_file():
        emotion_net = wsd.RegressionWisard(emotion_json)
    else:
        emotion_net = wsd.RegressionWisard(param)
    net.append(emotion_net)
    return net


if __name__ == "__main__":
    main(Path(args.input), Path(args.arousal_net), Path(args.valence_net), Path(args.emotion_net), Path(args.index_csv))
