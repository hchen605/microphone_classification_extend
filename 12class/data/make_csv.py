import os
import random
import pandas as pd

#data_dir = "/Users/hsin-hung.chen/Documents/HH document/ECE RA/CRISP/TIMIT/9m/"
data_dir = "/work/hchen605/recordings/9m/"
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
        train = dirp[7] == "TRAIN"
        type_3 = dirp[5][0]
        type_16 = dirp[5][:2]
        gender = dirp[9][0]

        if train: 
            coin = random.random()
            if coin <= 0.4:
                if gender == 'M':
                    m_val_list += [[os.path.join(dirpath, f), type_3, type_16]]
                else:
                    f_val_list += [[os.path.join(dirpath, f), type_3, type_16]]
                val_list += [[os.path.join(dirpath, f), type_3, type_16]]
            else:
                if gender == 'M':
                    m_tr_list += [[os.path.join(dirpath, f), type_3, type_16]]
                else:
                    f_tr_list += [[os.path.join(dirpath, f), type_3, type_16]]
                train_list += [[os.path.join(dirpath, f), type_3, type_16]]
        else:
            if gender == 'M':
                m_te_list += [[os.path.join(dirpath, f), type_3, type_16]]
            else:
                f_te_list += [[os.path.join(dirpath, f), type_3, type_16]]
            test_list += [[os.path.join(dirpath, f), type_3, type_16]]


train_csv = pd.DataFrame(train_list, columns=["filename", "3_types", "16_types"])
train_csv.to_csv("train_full_9m.csv", sep='\t', index=False)
test_csv = pd.DataFrame(val_list, columns=["filename", "3_types", "16_types"])
test_csv.to_csv("test_full_9m.csv", sep='\t', index=False)
val_csv = pd.DataFrame(val_list, columns=["filename", "3_types", "16_types"])
val_csv.to_csv("dev_full_9m.csv", sep='\t', index=False)

'''
m_tr_csv = pd.DataFrame(m_tr_list, columns=["filename", "3_types", "16_types"])
m_tr_csv.to_csv("train_male.csv", sep='\t', index=False)
m_te_csv = pd.DataFrame(m_te_list, columns=["filename", "3_types", "16_types"])
m_te_csv.to_csv("test_male.csv", sep='\t', index=False)
m_val_csv = pd.DataFrame(m_val_list, columns=["filename", "3_types", "16_types"])
m_val_csv.to_csv("dev_male.csv", sep='\t', index=False)

f_tr_csv = pd.DataFrame(f_tr_list, columns=["filename", "3_types", "16_types"])
f_tr_csv.to_csv("train_female.csv", sep='\t', index=False)
f_te_csv = pd.DataFrame(f_te_list, columns=["filename", "3_types", "16_types"])
f_te_csv.to_csv("test_female.csv", sep='\t', index=False)
f_val_csv = pd.DataFrame(f_val_list, columns=["filename", "3_types", "16_types"])
f_val_csv.to_csv("dev_female.csv", sep='\t', index=False)

'''
