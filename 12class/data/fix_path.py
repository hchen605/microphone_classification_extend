import pandas as pd
import os

data = pd.read_csv("utt2feat", sep='\t', encoding='ASCII')
file = data['filename'].tolist()
path = data['features'].tolist()
fixed_data = list()
for i,d in enumerate(file):
    p = path[i].split('/')
    p.insert(2,'train')
    fixed_path = '/'.join(p)
    print (fixed_path)
    fixed_data += [[d, fixed_path]]

utt2feat_file = pd.DataFrame(fixed_data, columns=["filename", "features"])       
utt2feat_file.to_csv("utt2feat", sep='\t', index=False)

