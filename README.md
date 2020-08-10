# PAMAP2-Loader

## About PAMAP2

[PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)

### センサの種類

- 気温
- 加速度(scale: Â±16g, resolution: 13-bit)
- 加速度(scale: Â±6g, resolution: 13-bit)
- ジャイロセンサ
- 磁気センサ

### 所持位置

手，腰，くるぶし

### 軸

x, y, z

## 想定するデータセットの構造

```

PAMAP2_Dataset + Optional + subject101.dat
                          + subject102.dat
                          :
                          + subject109.dat
               + Protocol
               + ...
               :
```

## 行動と対応するIDの一覧

| ID | Activities | 
|:---:|:-----:|
|1  | lying |
|2  | sitting |
|3  | standing |
|4  | walking |
|5  | running |
|7  | cycling |
|8  | nordic_walking |
|9  | watching_TV |
|10 | computer_work |
|11 | car_driving |
|12 | ascending stairs |
|13 | descending stairs |
|16 | vacuum_cleaning |
|17 | ironing |
|18 | folding_laundry |
|19 | house_cleaning |
|20 | playing_soccer |
|24 | rope_jumping |
|0  | other |

## Sample Code

```python
from pathlib import Path
from utils import *

# データセットのパスは"PAMAP2_Dataset/Protocol"を指定
ds_path = Path('...anywhere.../PAMAP2_Dataset/Protocol/')

# cache_dirには読み込んだデータのキャッシュを保存(デフォルトはカレントディレクトリ)
# 以降はキャッシュを使って高速に読み込むことが可能
pamap2 = PAMAP2(ds_path, cache_dir=Path('./'))

# ロード開始(初回はかなり時間がかかる)
pamap2.load()


# sliding-windowを作成 
## activitiesにはPAMAP2内で各行動に割り当てられたIDを指定
## [Return values]
## frames: sliding-windows
## labels: 行動ラベル
## person_labels: 被験者ラベル
## cid2act: 行動ラベルと実際の行動名の対応表
## pid2name: 被験者ラベルと実際の行動名の対応表

params = {
    'frame_size': 256,
    'activities': [1, 2, 3, 4, 5],
    'attributes': ['acc1'],
    'positions': ['chest'],
    'axes': ['x', 'y', 'z'], 
}

frames, labels, person_labels, cid2act, pid2name = pamap2.framing(**params)

```

