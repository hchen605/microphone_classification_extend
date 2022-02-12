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

test_list = list()

count = {}
test_paths = []
train_test_paths = []
validation_paths = []


# +

    
for (dirpath, dirnames, filenames) in os.walk(test_dir):
    #print(dirpath)
    #print(dirnames)
    #print(filenames)
    
    dirp = dirpath.split('/') 
    #train_paths = (set(train_paths) - set(validation_paths) - set(train_test_paths))
        
    for f in filenames:
        if not f.endswith(".wav"):
            continue
            
        test_paths.append(dirp[-1]+'/'+f)  
        
        #print(train_paths)
        
        
    


for f in test_paths:
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
    test_list += [[os.path.join(test_dir, f), label]]


# +


test_csv = pd.DataFrame(test_list, columns=["filename", "class"])
test_csv.to_csv("test_gsc.csv", sep='\t', index=False)

# -





