import os
import random
import pandas as pd

#data_dir = "/Users/hsin-hung.chen/Documents/HH document/ECE RA/CRISP/TIMIT/9m/"
data_dir = "/work/hchen605/recordings/"
train_list, val_list, test_list = list(), list(), list()
f_tr_list, f_val_list, f_te_list = list(), list(), list()
m_tr_list, m_val_list, m_te_list = list(), list(), list()
count = {}
dis = '9m'

for (dirpath, dirnames, filenames) in os.walk(data_dir):
    for f in filenames:
        if not f.endswith(".WAV"):
            continue
            
        dirp = dirpath.split('/')
        #print(dirp)
        train = dirp[5] == "TRAIN"
        type_3 = dirp[4][0]
        type_16 = dirp[4][:2]
        #gender = dirp[9][0]
        #dist = dirp[4]
        type_16 = 'small'
        
        if train: 
            coin = random.random()
            if coin <= 0.3:
                val_list += [[os.path.join(dirpath, f), type_3, type_16]]
            else:
                train_list += [[os.path.join(dirpath, f), type_3, type_16]]
        else:
            test_list += [[os.path.join(dirpath, f), type_3, type_16]]


train_csv = pd.DataFrame(train_list, columns=["filename", "3_types", "16_types"])
train_csv.to_csv("train_dimension_1.csv", sep='\t', index=False, mode='a')
test_csv = pd.DataFrame(val_list, columns=["filename", "3_types", "16_types"])
test_csv.to_csv("test_dimension_1.csv", sep='\t', index=False, mode='a')
val_csv = pd.DataFrame(val_list, columns=["filename", "3_types", "16_types"])
val_csv.to_csv("dev_dimension_1.csv", sep='\t', index=False, mode='a')


