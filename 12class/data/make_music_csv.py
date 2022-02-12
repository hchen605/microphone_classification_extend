import os
import random
import pandas as pd

data_dir = "/work/hchen605/music/"
train_list, val_list, test_list = list(), list(), list()
#f_tr_list, f_val_list, f_te_list = list(), list(), list()
#m_tr_list, m_val_list, m_te_list = list(), list(), list()
count = {}

for (dirpath, dirnames, filenames) in os.walk(data_dir):
    for f in filenames:
        if not f.endswith(".wav"):
            continue
            
        dirp = dirpath.split('/')
        #train = dirp[5] == "TRAIN"
        type_3 = dirp[4][0]
        type_16 = dirp[4][:2]
        #gender = dirp[7][0]
        #print(dirp)

        coin = random.random()
        if coin <= 0.2:
            val_list += [[os.path.join(dirpath, f), type_3, type_16]]
        elif 0.2 < coin <= 0.6:
            train_list += [[os.path.join(dirpath, f), type_3, type_16]]
        else:
            test_list += [[os.path.join(dirpath, f), type_3, type_16]]


train_csv = pd.DataFrame(train_list, columns=["filename", "3_types", "16_types"])
train_csv.to_csv("train_music_full.csv", sep='\t', index=False)
test_csv = pd.DataFrame(test_list, columns=["filename", "3_types", "16_types"])
test_csv.to_csv("test_music_full.csv", sep='\t', index=False)
val_csv = pd.DataFrame(val_list, columns=["filename", "3_types", "16_types"])
val_csv.to_csv("dev_music_full.csv", sep='\t', index=False)





