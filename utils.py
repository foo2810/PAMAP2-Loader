import numpy as np
import pandas as pd

import pickle
from pathlib import Path
from typing import Union, Optional

__all__ = ['PAMAP2'] + ['load_raw_data', 'seek_sensor_data', 'framing'] + ['ATTRIBUTES', 'POSITIONS', 'AXES', 'PERSONS', 'ACTIVITIES', 'columns']

ATTRIBUTES = ['temperature', 'acc1', 'acc2', 'gyro', 'mag']
POSITIONS = ['hand', 'chest', 'ankle']
AXES = ['x', 'y', 'z']

PERSONS = [
    'subject101', 'subject102', 'subject103',
    'subject104', 'subject105', 'subject106',
    'subject107', 'subject108', #'subject109',
]

ACTIVITIES = {
    1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running',
    7: 'cycling', 8: 'nordic_walking', 9: 'watching_TV', 10: 'computer_work',
    11: 'car_driving', 12: 'ascending stairs', 13: 'descending stairs',
    16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry',
    19: 'house_cleaning', 20: 'playing_soccer',
    24: 'rope_jumping',
    0: 'other',
}

columns = ['timestamp(s)', 'activity_id', 'heart_rate(bpm)']
for pos in POSITIONS:
    columns += ['IMU_{}_{}'.format(pos, ATTRIBUTES[0])]
    for attr in ATTRIBUTES[1:]:
        for axis in AXES:
            col = 'IMU_{}_{}_{}'.format(pos, attr, axis)
            columns += [col]
    columns += ['IMU_{}_orientation'.format(pos) for _ in range(4)]

def id2act(act_id):
    global ACTIVITIES
    return ACTIVITIES[act_id]

def load_raw_data(path):
    df = pd.read_csv(path, sep=' ', header=None)
    df.columns = columns
    df = df.fillna(method='ffill')
    return df

def seek_sensor_data2(seq_df):
    labels = np.array(seq_df['activity_id'])
    rshifted_labels = np.roll(labels, 1)

    diff = labels - rshifted_labels
    diff[0] = 1
    idxes = np.where(diff != 0)[0]

    segments = []
    for i in range(1, len(idxes)):
        segment = seq_df.iloc[idxes[i-1]:idxes[i]]
        segments += [segment]
    segments += [seq_df.iloc[idxes[-1]:]]

    # Check segment
    # flg = True
    # for seg in segments:
    #     seg_labels = np.array(seg['activity_id'])
    #     flg *= np.all(seg_labels == seg_labels[0])
    # if flg:
    #     print('OK')

    return segments

def seek_sensor_data(seq_df):
    label = seq_df['activity_id'].iloc[0]
    segments = []

    s_idx = 0
    for i in range(len(seq_df)):
        if seq_df['activity_id'].iloc[i] != label:
            segments += [seq_df.iloc[s_idx:i]]
            s_idx = i
            label = seq_df['activity_id'].iloc[i]
    
    return segments

def framing(segments, frame_size=256, activities=[1, 2, 3, 4, 5], attributes=['acc1'], positions=['chest'], axes=['x', 'y', 'z'], preprocesses=[]):
    frame_list = []
    label_list = []
    columns = []
    for pos in positions:
        for attr in attributes:
            for axis in axes:
                col = 'IMU_{}_{}_{}'.format(pos, attr, axis)
                columns += [col]
    act2cid = {}
    cid2act = {}
    for class_id, act in enumerate(activities):
        act2cid[act] = class_id
        cid2act[class_id] = id2act(act)

    for seg in segments:
        label = seg['activity_id'].iloc[0]

        if label not in activities:
            continue
       
        seg = seg[columns]
        class_id = act2cid[label]

        seg = np.array(seg)
        for preprocess in preprocesses:
            seg = preprocess(seg)

        frames = []
        for i in range(0, len(seg)-frame_size, frame_size):
            # frames += [np.array(seg.iloc[i:i+frame_size])[np.newaxis, ...]]
            frames += [seg[i:i+frame_size][np.newaxis, ...]]
        frames = np.concatenate(frames)
        labels = np.array([class_id for _ in range(len(frames))])

        frame_list += [frames]
        label_list += [labels]
    
    if len(frame_list) != 0:
        frame_list = np.concatenate(frame_list)
        label_list = np.concatenate(label_list)
    else:
        raise RuntimeError('No frame')

    return frame_list, label_list, cid2act


class PAMAP2:
    def __init__(self, dataset_path:Union[str,Path], cache_dir:Union[str,Path]=Path('./')):
        if type(dataset_path) is str:
            dataset_path = Path(dataset_path)
        if type(cache_dir) is str:
            cache_dir = Path(cache_dir)
        self.ds_path = dataset_path
        self.cache_dir = cache_dir

        self.is_loaded_raw_data = False
    
    # load from raw dataset and segmentation
    def load(self):
        self.segments = {}
        for person in PERSONS:
            print(' >>> loading ({})...'.format(person), end='', flush=True)
            fpath = self.ds_path / (person + '.dat')
            cache_path = self.cache_dir / 'segments_{}.pkl'.format(person)

            if cache_path.exists():
                with open(str(cache_path), 'rb') as fp:
                    self.segments[person] = pickle.load(fp)

            else:
                df = load_raw_data(str(fpath))
                self.segments[person] = seek_sensor_data2(df)
                save_path = self.cache_dir / 'segments_{}.pkl'.format(person)

                with open(str(save_path), 'wb') as fp:
                    pickle.dump(self.segments[person], fp, protocol=4)
            print('done')
            
        self.is_loaded_raw_data = True
    
    def framing(self, frame_size:int=256, persons:Optional[list]=None, activities:list=[1, 2, 3, 4, 5], attributes:list=['acc1'], positions:list=['chest'], axes:list=['x', 'y', 'z'], preprocesses:list=[]) -> tuple:
        """行動データのフレーム分けを行う

        Parameters
        ----------
        frame_size: int
            data width to split.
        persons: Option[list]
            frames were made from people in this list.
        activities:
            frames were made from activities in this list.
        attributes:
            frames were made from attributes in this list.
        positions:
            frames were made from positions in this list.
        axes:
            frames were made from axes in this list.
        preprocesses:
            a list of preprocessing functions.

        Returns
        -------
        frame_list:
            splited sensor frame data
        act_label_list:
            activities label list
        person_label_list:
            person label list
        cid2act:
            dictionary (key is id, value is activity string)
        pid2name:
            dictionary (key is id, value is person name)
        """
        if not self.is_loaded_raw_data:
            self.load()
        
        if persons is None:
            persons = globals()['PERSONS']

        frame_list = []
        act_label_list = []
        person_label_list = []
        pid2name = {}
        for num, person in enumerate(persons):
            frames, labels, cid2act = framing(self.segments[person], frame_size, activities, attributes, positions, axes, preprocesses)
            person_labels =  np.array([num for _ in range(len(frames))])
            pid2name[num] = person
            frame_list += [frames]
            act_label_list += [labels]
            person_label_list += [person_labels]
        frame_list = np.concatenate(frame_list)
        act_label_list = np.concatenate(act_label_list)
        person_label_list = np.concatenate(person_label_list)

        return frame_list, act_label_list, person_label_list, cid2act, pid2name
    

if __name__ == '__main__':
    ds_path = Path('path/to/dataset')
    pamap2 = PAMAP2(ds_path, cache_dir=Path('./'))
    pamap2.load()

    from preprocessing import lpf, hpf, bpf
    low_pass_filter_fn = lambda x: lpf(x, 10, 100)
    high_pass_filter_fn = lambda x: hpf(x, 10, 100)

    params = {
        'frame_size': 256,
        'activities': [1, 2, 3, 4, 5],
        'attributes': ['acc1'],
        'positions': ['chest'],
        'axes': ['x', 'y', 'z'], 
        'preprocesses': [low_pass_filter_fn],
    }

    frames, labels, person_labels, cid2act, pid2name = pamap2.framing(**params)
