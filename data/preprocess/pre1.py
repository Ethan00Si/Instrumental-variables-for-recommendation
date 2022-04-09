import pandas as pd
from tqdm import tqdm
import csv


folder_name = 'MINDlarge_train'
behaviors = pd.read_csv(folder_name+"/behaviors.tsv",sep='\t')
behaviors = behaviors.dropna(axis=0) #drop users without history
behaviors = behaviors.to_numpy()


import json
ID2idx = open('ID2idx.json','r')
ID2idx = json.load(ID2idx)

padding_length = 50 # truncate user browsing history at 50

fout = open('./train.csv','w')
# fout.write('user,item,ctr\n')
for line in tqdm(behaviors):
    user = line[3].split(' ')
    user = [ID2idx[x] for x in user]
    if len(user)>padding_length:
        user = user[:padding_length]
    else:
        for i in range(padding_length-len(user)):
            user.append(0)
    iteract_items = line[4].split(' ')
    iteract_items = [item.split('-') for item in iteract_items]
    for iteract in iteract_items:
        # fout.write(str(user))
        for item in user:
            fout.write(str(item)+',')
        # fout.write(',')
        fout.write(str(ID2idx[iteract[0]]))
        fout.write(',')
        fout.write(str(iteract[1]))
        fout.write('\n')
# dataset.to_csv('./train.tsv',sep='\t')