import numpy as np
import pandas as pd

__all__ = ['load_raw_data', 'seek_sensor_data', 'framing'] + ['attributes', 'positions', 'axes', 'persons', 'activities', 'columns']

attributes = ['temperature', 'acc1', 'acc2', 'gyro', 'mag']
positions = ['hand', 'chest', 'ankle']
axes = ['x', 'y', 'z']

persons = [
    'subject101', 'subject102', 'subject103',
    'subject104', 'subject105', 'subject106',
    'subject107', 'subject108', #'subject109',
]

activities = {
    1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running',
    7: 'cycling', 8: 'nordic_walking', 9: 'watching_TV', 10: 'computer_work',
    11: 'car_driving', 12: 'ascending stairs', 13: 'descending stairs',
    16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry',
    19: 'house_cleaning', 20: 'playing_soccer',
    24: 'rope_jumping',
    0: 'other',
}

columns = ['timestamp(s)', 'activity_id', 'heart_rate(bpm)']
for pos in positions:
    columns += ['IMU_{}_{}'.format(pos, attributes[0])]
    for attr in attributes[1:]:
        for axis in axes:
            col = 'IMU_{}_{}_{}'.format(pos, attr, axis)
            columns += [col]
    columns += ['IMU_{}_orientation'.format(pos) for _ in range(4)]

def load_raw_data(path):
    df = pd.read_csv(path, sep=' ', header=None)
    df.columns = columns
    return df

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

def framing(segments, frame_size=256, activities=[1, 2, 3, 4, 5], attributes=['acc1'], positions=['chest'], axes=['x', 'y', 'z']):
    frame_list = []
    label_list = []
    for seg in segments:
        label = seg['activity_id'].iloc[0]

        if label not in activities:
            continue

        columns = []
        for pos in positions:
            for attr in attributes:
                for axis in axes:
                    col = 'IMU_{}_{}_{}'.format(pos, attr, axis)
                    columns += [col]
        
        seg = seg[columns]

        frames = []
        for i in range(0, len(seg)-frame_size, frame_size):
            frames += [np.array(seg.iloc[i:i+frame_size])[np.newaxis, ...]]
        frames = np.concatenate(frames)
        labels = np.array([label for _ in range(len(frames))])

        frame_list += [frames]
        label_list += [labels]
    
    if len(frame_list) != 0:
        frame_list = np.concatenate(frame_list)
        label_list = np.concatenate(label_list)
    else:
        raise RuntimeError('No frame')

    return frame_list, label_list

