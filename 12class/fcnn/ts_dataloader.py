import librosa
import numpy as np
import pandas as pd
import soundfile as sound

from tensorflow import keras

import glob
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Pool

sr = 16000
duration = 3

# +
def _load_data(data):
    wav, y_3, y_16 = data
    stereo, fs = sound.read(wav)
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)
    assert stereo.shape[0] > 16000
    if stereo.shape[0] > sr*duration:
        start = (stereo.shape[0] - sr*duration) // 2
        x = stereo[start:start+sr*duration]
    else:
        x = np.pad(stereo, (0, sr*duration-stereo.shape[0]))
    
    return x, y_3, y_16

def load_data(data_csv):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels_3 = data_df['3_types'].to_list()
    labels_16 = data_df['16_types'].to_list()
    datas = zip(wavpath, labels_3, labels_16)

    with Pool(32) as p:
        return p.map(_load_data, datas)
    
def _load_data_gsc(data):
    wav, y = data
    #print(y)
    stereo, fs = sound.read(wav)
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)
    #assert stereo.shape[0] > 16000
    if stereo.shape[0] > sr:
        start = 0
        x = stereo[start:start+sr]
    else:
        x = np.pad(stereo, (0, sr-stereo.shape[0]))
    
    #print(x)
    return x, y

def load_data_gsc(data_csv):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    label = data_df['class'].to_list()
    datas = zip(wavpath, label)

    with Pool(32) as p:
        return p.map(_load_data_gsc, datas)
# -


