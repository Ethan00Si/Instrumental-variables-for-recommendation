import torch.nn as nn
import torch

from .fc import FullyConnectedLayer


'''
Refered implementation: https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/din.py
'''

class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], bias=[True, True], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size
        
        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(1), torch.tensor(0))
        

        # multiply weight
        output = torch.matmul(attention_score, user_behavior)

        return output
        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        
        queries = query.expand(-1, user_behavior_len, -1)
        
        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior],
             dim=-1) # as the source code, subtraction simulates verctors' difference
        
        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output) # [B, T, 1]

        return attention_score

if __name__ == "__main__":
    a = AttentionSequencePoolingLayer()
    
    import torch
    b = torch.zeros((3, 1, 4))
    c = torch.zeros((3, 20, 4))
    d = torch.ones((3, 1))
    a(b, c, d)