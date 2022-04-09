import torch.nn as nn
import torch

from .modules.fc import FullyConnectedLayer
from .modules.attention_pooling_layer import AttentionSequencePoolingLayer

from .modules.basic_model import BasicModel
from data.loader import  load_pretrained_embedding, load_corresponding_query
import torchsnooper

embedding_size = 768


class DeepInterestNetwork(BasicModel):
    def __init__(self, config):
        '''
            Deep Interest Network for CTR prediction
            Refered implementation: https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/din.py
        '''
        super().__init__(config)
        self.name = 'din_base'
        self.device = config.device

        _, photo_embedding_matrix = load_pretrained_embedding()

        self.photo_embedding_layer = nn.Embedding.from_pretrained(photo_embedding_matrix, freeze=True)
        
        self.attn = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.fc_layer = FullyConnectedLayer(input_size=2*embedding_size,
                                            hidden_unit=[200, 80, 1],
                                            bias=[True, True, False],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='dice',
                                            dice_dim=3)


    def forward(self, x):
        # user_features -> dict (key:feature name, value: feature tensor)
        browse_photo, photo = x
        browse_photo, photo = browse_photo.to(self.device), photo.to(self.device)
       
        browse_embedding = self.photo_embedding_layer(browse_photo)
        browse_mask = torch.where(browse_photo==0, 1, 0).bool()

        photo_embedded = self.photo_embedding_layer(photo)

        browse_atten = self.attn(photo_embedded,
                            browse_embedding, browse_mask) 
        concat_feature = torch.cat([photo_embedded, browse_atten], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return output.squeeze(1).squeeze(1)


class DIN_on_KuaiShou(BasicModel):
    '''
        We conducted experiments on KuaiShou dataset with this model.
        Note that it was adapted to Kuaishou dataset by adding queries and clicked items in the search history 
        as additional history of user behaviors.
    '''
    def __init__(self, config):
        super().__init__(config)
        self.name = 'din_base'
        self.device = config.device

        query_embedding_matrix, item_embedding_matrix = load_pretrained_embedding()

        self.item_embedding_layer = nn.Embedding.from_pretrained(item_embedding_matrix, freeze=False)
        self.query_embedding_layer = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True)
        
        self.attn1 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.attn2 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.attn3 = AttentionSequencePoolingLayer(embedding_dim=embedding_size)
        self.fc_layer = FullyConnectedLayer(input_size=4*embedding_size,
                                            hidden_unit=[200, 80, 1],
                                            bias=[True, True, False],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='dice',
                                            dice_dim=3)


    def forward(self, x):
        x = [i.to(self.device) for i in x]
        search_query, browse_photo, search_click_photo, photo = x


        #batch_size,history_length,feature
        query_embedding = self.query_embedding_layer(search_query)
        query_mask = torch.where(search_query==0, 1, 0).bool()

        click_embedding = self.item_embedding_layer(search_click_photo)
        search_click_mask = torch.where(search_click_photo==0, 1, 0).bool()
       
        browse_embedding = self.item_embedding_layer(browse_photo)
        browse_mask = torch.where(browse_photo==0, 1, 0).bool()
        
        item_embedded = []

        item_embedded = self.item_embedding_layer(photo)

        query_atten = self.attn1(item_embedded, 
                            query_embedding,
                            query_mask)
        click_atten = self.attn2(item_embedded,
                            click_embedding, search_click_mask)
        browse_atten = self.attn3(item_embedded,
                            browse_embedding, browse_mask) 
        concat_feature = torch.cat([item_embedded, query_atten, click_atten, browse_atten], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return output.squeeze(1).squeeze(1)
