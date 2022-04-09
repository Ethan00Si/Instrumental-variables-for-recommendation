import torch.nn as nn
import torch

from .modules.fc import FullyConnectedLayer
from .modules.attention_pooling_layer import AttentionSequencePoolingLayer

from .modules.basic_model import BasicModel
from data.loader import  load_pretrained_embedding,load_corresponding_query
from .modules.phi import MLP
import torchsnooper

embedding_size = 768


class DeepInterestNetwork(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'din_IV_mlp'
        self.device = config.device

        query_embedding_matrix, photo_embedding_matrix = load_pretrained_embedding()

        self.photo_embedding_layer = nn.Embedding.from_pretrained(photo_embedding_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True)
        
        corresponding_query_matrix, cor_query_pinv = load_corresponding_query()
        self.corresponding_query_embedding = nn.Embedding.from_pretrained(corresponding_query_matrix, freeze=True)
        #Since query embedding is fixed, we can calculate pseudoinverse of corresponding query embedding matrix offline to accelerate training
        self.cor_query_pinv = nn.Embedding.from_pretrained(cor_query_pinv, freeze=True)

        #MLP0
        self.phi = MLP(config.phi)


        self.add_regularization_weight(self.phi.parameters(), l2=config.l2_lambda)

        #MLP1 and MLP2
        self.alpha = MLP(config.alpha)
        self.add_regularization_weight(self.alpha.parameters(), l2=config.l2_lambda)
        self.beta = MLP(config.beta)
        self.add_regularization_weight(self.beta.parameters(), l2=config.l2_lambda)

        self.attn1 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.fc_layer = FullyConnectedLayer(input_size=2*embedding_size,
                                            hidden_unit=[200, 80, 1],
                                            bias=[True, True, False],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='dice',
                                            dice_dim=3)

        self.prob_sigmoid = nn.Sigmoid()

        

    def iv(self, photo, query, query_pinv):
        query_origin = query.flatten(start_dim=2)
        # batch,len,1
        alpha = self.prob_sigmoid(self.alpha(torch.cat([photo,query_origin],dim=-1)))
        beta = self.prob_sigmoid(self.beta(torch.cat([photo,query_origin],dim=-1)))
        #batch, len, feature, 1
        photo = photo.unsqueeze(dim=-1)
                        #batch,len,feature,10        #batch,len,10,feature #batch,len,feature,1
        t_1 = torch.matmul(query, torch.matmul(query_pinv, photo))
        
        q_t = torch.matmul(photo, alpha.unsqueeze(dim=-1))+\
            torch.matmul(t_1, beta.unsqueeze(dim=-1))
        return q_t.squeeze_(dim=-1)#batch,len,feature


    def forward(self, x):
        browse_photo, photo = x
        browse_photo, photo = browse_photo.to(self.device),\
                photo.to(self.device)

        query_num_per_photo = 1

        photo_embedding = self.photo_embedding_layer(photo)#batch,1,feature
        cor_query_embedding = self.corresponding_query_embedding(photo).reshape(photo.shape[0],1,embedding_size,query_num_per_photo)
        cor_query_pinv = self.cor_query_pinv(photo).reshape(photo.shape[0],1,query_num_per_photo,embedding_size)
        photo_embedding = self.phi(photo_embedding) #batch, 1, feature
        photo_embedding = self.iv(photo_embedding, cor_query_embedding, cor_query_pinv) #batch, 1, feature
        

        browse_embedding = self.photo_embedding_layer(browse_photo)
        browse_cor_query_embedding = self.corresponding_query_embedding(browse_photo).reshape(browse_photo.shape[0], browse_photo.shape[1], embedding_size, query_num_per_photo)
        browse_cor_query_pinv = self.cor_query_pinv(browse_photo).reshape(browse_photo.shape[0], browse_photo.shape[1], query_num_per_photo, embedding_size)
        browse_embedding = self.phi(browse_embedding)#batch, len, feature
        browse_embedding = self.iv(browse_embedding, browse_cor_query_embedding, browse_cor_query_pinv)#batch, len, feature
        browse_mask = torch.where(browse_photo==0, 1, 0).bool()
        


        browse_atten = self.attn1(photo_embedding,
                            browse_embedding, browse_mask) 
        concat_feature = torch.cat([photo_embedding, browse_atten], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return output.squeeze(1).squeeze(1)


if __name__ == "__main__":
    a = DeepInterestNetwork()



