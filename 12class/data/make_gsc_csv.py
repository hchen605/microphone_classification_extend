import os
import random
import pandas as pd

# +
data_dir = "/work/hchen605/speech_commands/"
train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'
noise_dir = train_dir + '_background_noise_/'
WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
SILENCE = '_silence_'
UNKNOWN = '_unknown_'
BACKGROUND_NOISE = '_background_noise_'

train_list, val_list, test_list = list(), list(), list()
f_tr_list, f_val_list, f_te_list = list(), list(), list()
m_tr_list, m_val_list, m_te_list = list(), list(), list()
count = {}
train_paths = []
train_test_paths = []
validation_paths = []


# +
#if 'testing_list.txt' in filenames:
with open(train_dir+'testing_list.txt') as f:
    feats = f.readlines()
    #print(feats)
for feat in feats:
    train_test_paths.append(feat[:-1])
print(train_test_paths[0])

with open(train_dir+'validation_list.txt') as f:
    feats = f.readlines()
    #print(feats)
for feat in feats:
    validation_paths.append(feat[:-1])
print(validation_paths[0])
    
for (dirpath, dirnames, filenames) in os.walk(train_dir):
    #print(dirpath)
    #print(dirnames)
    #print(filenames)
    
    dirp = dirpath.split('/') 
    #train_paths = (set(train_paths) - set(validation_paths) - set(train_test_paths))
        
    for f in filenames:
        if not f.endswith(".wav"):
            continue
            
        train_paths.append(dirp[-1]+'/'+f)  
        
        #print(train_paths)
        
        
        
#print(train_paths[:20])
#print(len(train_paths))
#print(len(validation_paths))
#print(len(train_test_paths))
train_paths = list(set(train_paths) - set(validation_paths) - set(train_test_paths))
print(len(train_paths))
print(len(validation_paths))
print(len(train_test_paths))
#print(train_paths[:3])
#print(validation_paths[:2])
#print(train_paths[:2])


for f in train_paths:
    #print(f)
    word = f.split('/')[0]
    #print(word)
    if word in WORDS:
        label = word
    elif word == SILENCE or word == BACKGROUND_NOISE:
        # The main tar file already contains all of the test files, except for
        # the silence ones. In fact it does not contain silence files at all.
        # So for the test set we take the silence files from the test tar file,
        # while for train and validation we build them from the
        # _background_noise_ folder.
        label = SILENCE
    else:
        # Note that in the train and validation there are a lot more _unknown_
        # labels than any of the other ones.
        label = UNKNOWN
    #print(label)
    train_list += [[os.path.join(train_dir, f), label]]

for f in validation_paths:
    #print(f)
    word = f.split('/')[0]
    print(word)
    if word in WORDS:
        label = word
    elif word == SILENCE or word == BACKGROUND_NOISE:
        # The main tar file already contains all of the test files, except for
        # the silence ones. In fact it does not contain silence files at all.
        # So for the test set we take the silence files from the test tar file,
        # while for train and validation we build them from the
        # _background_noise_ folder.
        label = SILENCE
    else:
        # Note that in the train and validation there are a lot more _unknown_
        # labels than any of the other ones.
        label = UNKNOWN
    #print(label)
    val_list += [[os.path.join(train_dir, f), label]]

# -

train_csv = pd.DataFrame(train_list, columns=["filename", "class"])
train_csv.to_csv("train_gsc.csv", sep='\t', index=False)
val_csv = pd.DataFrame(val_list, columns=["filename", "class"])
val_csv.to_csv("dev_gsc.csv", sep='\t', index=False)
'''
test_csv = pd.DataFrame(test_list, columns=["filename", "3_types", "16_types"])
test_csv.to_csv("test_full.csv", sep='\t', index=False)
val_csv = pd.DataFrame(val_list, columns=["filename", "3_types", "16_types"])
val_csv.to_csv("dev_full.csv", sep='\t', index=False)
'''





