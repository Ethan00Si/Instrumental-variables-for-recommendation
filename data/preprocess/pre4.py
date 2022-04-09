import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from tqdm import tqdm
import math
import torchsnooper

model = AutoModel.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',do_lower_case=True)

model = model.to('cuda:0')

import pandas as pd
news = pd.read_csv('news_embedding_query.tsv',sep='\t')
embedding = news['embedding'].to_numpy() #generate item embedding

# embedding = news['query'].to_numpy() #generate query embedding
'''
Please note that text embeddings of news articles were used as item embeddings on experiments 
of MIND dataset. In the field of news recommendation, it's widely known that using pre-trained 
text embeddings as item embeddings leads to better performance than randomly initialized item 
embeddings.
'''
 
import gc
embedding_vec = torch.zeros((embedding.shape[0]+1,768))
batch_size = 8
for i in tqdm(range(math.ceil(embedding.shape[0] / batch_size))):
    batch_input = embedding[i*batch_size:(i+1)*batch_size].tolist()
    tokens = tokenizer(batch_input,padding="max_length",truncation=True,return_tensors="pt",max_length=100)
    tokens = tokens.to('cuda:0')
    features = model(**tokens)
    features = features.last_hidden_state[:,0]
    embedding_vec[i*batch_size+1:(i+1)*batch_size+1] = features
    del features
    gc.collect()
    # print(features)
    # break

embedding_vec = embedding_vec.numpy()

import numpy as np
np.save('embedding_vec.npy', embedding_vec)
# np.save('query_vec.npy', embedding_vec) 

