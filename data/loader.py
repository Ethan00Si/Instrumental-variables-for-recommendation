import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.dataset import IterableDataset

class DatasetFromCSV(IterableDataset):
    def __init__(self, train=False, test=False, validation=False):
        '''
        data format: 
            * query history : (50,)
            * click history in search scenario: (50,) 
            * browse history in recommendation scenario: (50,) 
            * item (candidate item) : (1,)
            * label : (1,)
        '''
        if train:
            self.file_path = '/home/zihua_si/mind_data/train.csv'
            self.length = 81785129 
        elif test:
            self.file_path = '/home/zihua_si/mind_data/dev.csv'
            self.length = 13662442
        elif validation:
            pass
            '''Validation dataset of MIND was used as test set'''
        else:
            raise ValueError("incorrect dataset")


    def __len__(self):
        return self.length

    def __iter__(self):
        with open(self.file_path, 'r') as file_obj:
            for line in file_obj:
                line_data = line.strip('\n').split(',')
                line_data = torch.from_numpy(np.array(line_data, dtype='int'))
                # yield line_data
                # yield line_data[0:50],line_data[50:100],line_data[100:150],line_data[150:151],line_data[151].float()
                yield line_data[:50],line_data[50:51],line_data[51].float()



class TestDataset(IterableDataset):
    def __init__(self):
        self.file_path = '/home/zihua_si/mind_data/test.csv'
        self.length = 13662442

    def __len__(self):
        return self.length

    def __iter__(self):
        with open(self.file_path, 'r') as file_obj:
            for line in file_obj:
                line_data = line.strip('\n').split(',')
                line_data = torch.from_numpy(np.array(line_data, dtype='int'))
                user = line_data[:50]
                item = line_data[50:51]
                label = line_data[51].float()
                impression = line_data[52]
                yield user,item,label,impression.tolist()


def get_dataloader(batch_size, mode):
    '''
    generate data batches in stream

    Args:
        batch_size: size of batch
        mode: string, 'train' or 'test' or 'validation'
    
    Return:
        dataloader
    '''
    dataloader = None
    if mode == 'train':
        dataset = DatasetFromCSV(train=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, prefetch_factor=2,\
            shuffle=False, pin_memory=True, num_workers=0)
    elif mode == 'test' or mode == 'validation':
        dataset = TestDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, prefetch_factor=2,\
            shuffle=False, pin_memory=True, num_workers=0)

    return dataloader
    
def load_pretrained_embedding(root_path='/data/iv/new_data/'):
    '''
    Returns:
        query_embedding(np.array): (number of data, 64)
        photo_embedding(np.array): (number of data, 64)
    '''
    root_path = '/home/zihua_si/mind_data/'
    # root_path = './data/dataset/'
    query_embedding_matrix = np.load(root_path+'query_vec.npy',allow_pickle=True)
    pid_embedding_matrix = np.load(root_path+'embedding_vec.npy',allow_pickle=True)
    # pid_embedding_matrix = np.random.randn(pid_embedding_matrix.shape[0],64)
    # query_embedding_matrix = np.random.randn(query_embedding_matrix.shape[0],64)
    return torch.tensor(query_embedding_matrix).float(), torch.tensor(pid_embedding_matrix).float() 



def load_corresponding_query():
    corresponding_query = np.load('/home/zihua_si/mind_data/query_vec.npy',allow_pickle=True)
    cor_query_pinv = np.load('/home/zihua_si/mind_data/Zt_pinv.npy',allow_pickle=True)
    corresponding_query = torch.tensor(corresponding_query).float()
    cor_query_pinv = torch.tensor(cor_query_pinv).float()
    corresponding_query = corresponding_query.flatten(start_dim=1)
    cor_query_pinv = cor_query_pinv.flatten(start_dim=1)
    return corresponding_query, cor_query_pinv



    