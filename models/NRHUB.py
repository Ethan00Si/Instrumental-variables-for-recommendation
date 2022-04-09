from re import search
import torch
import torch.nn as nn
from .modules.basic_model import BasicModel
from data.loader import load_pretrained_embedding
from .modules.attention import Attention
import torchsnooper

class NRHUB(BasicModel):
    def __init__(self, config):
        '''
            Neural News Recommendation with Heterogeneous User Behavior
            Refered implementation: https://github.com/wuch15/NRHUB
            On the experiments of MIND dataset, it was adapted by removing the query encoder module and news encoder 
            module in user representation learning because MIND doesn’t support users’ search history.
        '''
        super().__init__(config=config)
        self.name = 'NRHUB'
        
        query_embedding_matrix, photo_embedding_matrix = load_pretrained_embedding()

        self.photo_embedding_layer = nn.Embedding.from_pretrained(photo_embedding_matrix, freeze=True)

        Dense_dim = 200
        photo_feature_dim = 768
        padding_len = 50
        self.browse_photo_att = Attention(padding_len, photo_feature_dim, Dense_dim)

        self.photo_rep = nn.Sequential(
            nn.Linear(photo_feature_dim, Dense_dim),
            nn.Tanh()
        )

        self.user_rep = nn.Sequential(
           nn.Linear(photo_feature_dim, Dense_dim),
           nn.Tanh()
        )

        self.prob_sigmoid = nn.Sigmoid()

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    # @torchsnooper.snoop()
    def forward(self, x):
        browse_photo, photo = x
        browse_photo = browse_photo.to(self.device)
        photo = photo.to(self.device)

        photo_embedding = self.photo_embedding_layer(photo) #batch,1,feature
        photo_rep = self.photo_rep(photo_embedding) #batch,1,dense


        browse_embedding = self.photo_embedding_layer(browse_photo)
        browse_mask = torch.where(browse_photo==0, 1, 0).bool()
        browse_rep = self.browse_photo_att(browse_embedding, browse_mask)

        user_rep = self.user_rep(browse_rep) #batch,dense
        user_rep = user_rep.unsqueeze(dim=-1)#batch,dense,1

        logits = torch.matmul(photo_rep,user_rep)#batch,1,1
        prob = self.prob_sigmoid(logits).squeeze(-1).squeeze(-1)
        
        return prob




class NRHUB_on_Kuaishou(BasicModel):
    def __init__(self, config):
        '''
            We conduct experiments on KuaiShou dataset using this model.
        '''
        super().__init__(config=config)
        self.name = 'NRHUB'
        
        query_embedding_matrix, photo_embedding_matrix = load_pretrained_embedding()
        query_embedding_matrix = query_embedding_matrix.to(self.device)
        photo_embedding_matrix = photo_embedding_matrix.to(self.device)

        self.photo_embedding_layer = nn.Embedding.from_pretrained(photo_embedding_matrix, freeze=True)
        self.query_embedding_layer = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True)

        Dense_dim = 200
        photo_feature_dim = 64
        query_feature_dim = 64
        padding_len = 50
        self.search_query_att = Attention(padding_len, query_feature_dim, Dense_dim)
        self.search_click_att = Attention(padding_len, photo_feature_dim, Dense_dim)
        self.browse_photo_att = Attention(padding_len, photo_feature_dim, Dense_dim)

        self.photo_rep = nn.Sequential(
            nn.Linear(photo_feature_dim, Dense_dim),
            nn.Tanh()
        )

        self.user_rep = nn.Sequential(
           nn.Linear(query_feature_dim, Dense_dim),
           nn.Tanh(),
           Attention(3, Dense_dim, 100),
        )

        self.prob_sigmoid = nn.Sigmoid()

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    # @torchsnooper.snoop()
    def forward(self, x):
        search_query, browse_photo, search_click_photo, photo = x
        search_query = search_query.to(self.device)
        browse_photo = browse_photo.to(self.device)
        search_click_photo = search_click_photo.to(self.device)
        photo = photo.to(self.device)

        photo_embedding = self.photo_embedding_layer(photo) #batch,1,feature
        photo_rep = self.photo_rep(photo_embedding) #batch,1,dense

        #batch_size,padding_len,feature
        query_embedding = self.query_embedding_layer(search_query)
        query_mask = torch.where(search_query==0, 1, 0).bool()
        #batch_size,padding_len,feature -> batch_size,feature
        query_rep = self.search_query_att(query_embedding, query_mask)

        click_embedding = self.photo_embedding_layer(search_click_photo)
        search_click_mask = torch.where(search_click_photo==0, 1, 0).bool()
        click_rep = self.search_click_att(click_embedding, search_click_mask)

        browse_embedding = self.photo_embedding_layer(browse_photo)
        browse_mask = torch.where(browse_photo==0, 1, 0).bool()
        browse_rep = self.browse_photo_att(browse_embedding, browse_mask)

        #batch,3,feature
        query_rep.unsqueeze_(dim=1)
        click_rep.unsqueeze_(dim=1)
        browse_rep.unsqueeze_(dim=1)
        user_rep = torch.cat((query_rep,click_rep,browse_rep),dim=1)
        user_rep = self.user_rep(user_rep) #batch,dense
        user_rep = user_rep.unsqueeze(dim=-1)#batch,dense,1

        logits = torch.matmul(photo_rep,user_rep)#batch,1,1
        prob = self.prob_sigmoid(logits).squeeze(-1).squeeze(-1)
        
        return prob

