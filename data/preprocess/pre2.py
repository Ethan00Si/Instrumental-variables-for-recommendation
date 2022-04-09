from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            if current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
            else:
                    continue
    return continuous_chunk

'''
if entity is missing, generate by nltk. else concate all entity words 
'''

import pandas as pd
news = pd.read_csv('/Users/ethanxi/Desktop/mind/news_all.tsv', sep='\t')

news_entity = news.copy()
news_entity['total_entities'] = ''
news = news.to_numpy()


# fout = open('./news_with_entity.tsv','a+')

for idx in range(len(news)):
    print(idx,end='\r')
    if news[idx][6] == '[]' and news[idx][7]=='[]':
        if type(news[idx][4])==float:
            '''news without abstract'''
            tit_abs = news[idx][3]
        else:
            tit_abs = news[idx][3]+' '+news[idx][4]
        # print(tit_abs)
        entity = get_continuous_chunks(tit_abs)
        # print(entity)
        # print(idx)
        total_entities = ' '.join(entity)
        news_entity.loc[idx,'total_entities'] = total_entities
        # break
    else:
        total_entities = []
        if news[idx][6] != '[]' and type(news[idx][6])!=float:
            try:
                total_entities.append([x['Label'] for x in eval(news[idx][6])])
            except:
                print(idx)
        if news[idx][7] != '[]' and type(news[idx][7])!=float:
            try:
                total_entities.append([x['Label'] for x in eval(news[idx][7])])
            except:
                print(idx)
        total_entities = [item for sub_list in total_entities for item in sub_list]
        total_entities = ' '.join(str(e) for e in total_entities)
        news_entity.loc[idx,'total_entities'] = total_entities
news_entity.to_csv('./news_with_entity.tsv',sep='\t')