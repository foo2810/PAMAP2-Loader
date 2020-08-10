from pathlib import Path
from utils import *
from preprocessing import lpf, hpf, bpf

ds_path = Path('../PAMAP2_Dataset/Protocol/')
pamap2 = PAMAP2(ds_path, cache_dir=Path('./'))
pamap2.load()

# low_pass_filter_fn = lambda x: lpf(x, 10, 100)
# high_pass_filter_fn = lambda x: hpf(x, 10, 100)

params = {
    'frame_size': 256,
    'activities': [1, 2, 3, 4, 5],
    'attributes': ['acc1'],
    'positions': ['chest'],
    'axes': ['x', 'y', 'z'], 
    'preprocesses': [low_pass_filter_fn],
}

frames, labels, person_labels, cid2act, pid2name = pamap2.framing(**params)
