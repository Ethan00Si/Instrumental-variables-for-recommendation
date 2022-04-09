import numpy as np
from tqdm import tqdm
import math
import ast
import torch

query = np.load("query_vec.npy",allow_pickle=True)
# query_vec.npy is Zt
query = torch.Tensor(query)


Zt = query.unsqueeze(-1)#np.zeros((query_index.shape[0], 64, top_k))
Zt_pinv = np.zeros((query.shape[0],  1, 768))

      
for i in tqdm(range(Zt.shape[0])):
    Zt_pinv[i] = np.linalg.pinv(Zt[i])

np.save("./Zt_pinv.npy", Zt_pinv)