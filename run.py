from pathlib import Path

import numpy as np
import pandas as pd

import pickle

from utils import *

fpath_base = Path('../PAMAP2_Dataset/Protocol/')

for person in persons:
    fpath = fpath_base / (person + '.dat')
    print(fpath)

    df = load_raw_data(str(fpath))
    df.to_csv('pamap2_{}.csv'.format(person))

    segments = seek_sensor_data(df)
    # with open('segments_{}.pkl'.format(person), 'rb') as fp:
    #     segments = pickle.load(fp)

    # Check segments
    flg = False
    for seg in segments:
        label = seg['activity_id'].iloc[0]
        for i in range(len(seg)):
            if seg['activity_id'].iloc[i] != label:
                print(' >>> Miss: {} - {}'.format(label, seg['activity_id'].iloc[i]))
                flg = True
                break
    if not flg:
        print('Segmentation: OK')

    with open('segments_{}.pkl'.format(person), 'wb') as fp:
        pickle.dump(segments, fp, protocol=4)

    # Make sliding-window
    frames, labels = framing(segments, frame_size=256)

    print('frames: {}'.format(frames.shape))
    print('labels: {}'.format(labels.shape))

    with open('frames_{}.pkl'.format(person)) as fp:
        pickle.dump(frames, fp, protocol=4)

