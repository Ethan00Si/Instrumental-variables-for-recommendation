import torch.nn as nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, signal_length, hidden_dim, Dense_dim):
        super().__init__()
        self.signal_length = signal_length
        self.hidden_dim = hidden_dim
        self.dense_dim = Dense_dim
        self.Uq = nn.Linear(self.hidden_dim, self.dense_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.vq = nn.Parameter(torch.randn(1,1,self.dense_dim))

    def reset_parameters(self):
        nn.init.normal_(self.vq)
        nn.init.normal_(self.Uq)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [bs, sl, hd]
        Return
            res: [bs, hd]
        """
        # bs, sl ,dd
        key = self.tanh(self.Uq(x))

        # bs, 1, sl
        score = self.vq.matmul(key.transpose(-2,-1))
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1), torch.tensor(-1e6))
        
        weight = self.softmax(score)
        res = weight.matmul(x).sum(dim=1)
        # bs, hd
        return res

if __name__ == '__main__':
    att = Attention(2, 5, 10)#feature, hid
    # input_data = torch.randn((4,2))
    # a = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0],[7.0,8.0]]).unsqueeze(dim=1)
    mask = torch.zeros((4,2))
    mask[0,0] = 1
    mask[1,1] = 1
    mask[2,0] = 1
    mask = mask.unsqueeze(-2)

    # print(a.shape) #batch,padding_len,feature
    a = torch.randn((4,2,5))
    print(att(a,mask))
    # print(att(a,mask))
        