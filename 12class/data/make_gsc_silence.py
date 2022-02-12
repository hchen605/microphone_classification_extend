import os
import random
import pandas as pd
import soundfile as sound
#from tensorflow_datasets.core import lazy_imports_lib
#import tensorflow_datasets.public_api as tfds

data_dir = "/work/hchen605/speech_commands/"
sr = 16000
SAMPLE_RATE = 16000
BACKGROUND_NOISE = '_background_noise_'
train_list, val_list, test_list = list(), list(), list()
f_tr_list, f_val_list, f_te_list = list(), list(), list()
m_tr_list, m_val_list, m_te_list = list(), list(), list()
count = {}

# +
train_dir = data_dir + 'train/'
noise_dir = train_dir + '_background_noise_/'

for (dirpath, dirnames, filenames) in os.walk(noise_dir):
    #relpath, wavname = os.path.split(path)
    #_, word = os.path.split(relpath)
    print(dirpath)
    print(dirnames)
    print(filenames)
    for f in filenames:
        #if word == BACKGROUND_NOISE:
        # Special handling of background noise. We need to cut these files to
        # many small files with 1 seconds length, and transform it to silence.
        #audio_samples = np.array(
        #    lazy_imports_lib.lazy_imports.pydub.AudioSegment.from_file(
        #        file_obj, format='wav').get_array_of_samples())
        if f[-3:] =='wav':
            audio_samples, fs = sound.read(dirpath+f)
            #print(fs)
            for start in range(0,
                               len(audio_samples) - SAMPLE_RATE, SAMPLE_RATE // 2):
                audio_segment = audio_samples[start:start + SAMPLE_RATE]
                cur_id = '{}_{}'.format(f[:-4], start)
                #print(cur_id)
                #example = {'audio': audio_segment, 'label': label}
                sound.write(noise_dir+cur_id+'.wav', audio_segment, sr)
        
# -







