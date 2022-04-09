import torch
import torch.nn as nn

from .modules.basic_model import BasicModel
from data.loader import  load_pretrained_embedding, load_corresponding_query
from .modules.attention import Attention
from .modules.phi import MLP
import torchsnooper


class IV_nonlinear(BasicModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.name = 'IV_phi_mlp'
        
        query_embedding_matrix, photo_embedding_matrix = load_pretrained_embedding()

        self.photo_embedding_layer = nn.Embedding.from_pretrained(photo_embedding_matrix, freeze=True)

        corresponding_query_matrix, cor_query_pinv = load_corresponding_query()
        #Since query embedding is fixed, we can calculate pseudoinverse of corresponding query embedding matrix offline to accelerate training
        self.corresponding_query_embedding = nn.Embedding.from_pretrained(corresponding_query_matrix, freeze=True)
        self.cor_query_pinv = nn.Embedding.from_pretrained(cor_query_pinv, freeze=True)

        #MLP0
        self.phi = MLP(config.phi)


        self.add_regularization_weight(self.phi.parameters(), l2=config.l2_lambda)


        #MLP1 and MLP2
        self.alpha = MLP(config.alpha)
        self.beta = MLP(config.beta)

        photo_feature_dim = 768
        Dense_dim = 200
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

        for m in self.children():
            if isinstance(m, nn.Embedding):
                continue
            m.to(self.device)

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

    # @torchsnooper.snoop()
    def forward(self, x):
        browse_photo, photo = x

        query_num_per_photo = 1#10

        photo_embedding = self.photo_embedding_layer(photo).to(self.device) #batch,1,feature
        cor_query_embedding = self.corresponding_query_embedding(photo).reshape(photo.shape[0],1,768,query_num_per_photo).to(self.device)
        cor_query_pinv = self.cor_query_pinv(photo).reshape(photo.shape[0],1,query_num_per_photo,768).to(self.device)
        photo_embedding = self.phi(photo_embedding) #batch, 1, feature
        photo_embedding = self.iv(photo_embedding, cor_query_embedding, cor_query_pinv) #batch, 1, feature
        photo_rep = self.photo_rep(photo_embedding) #batch,1,dense



        browse_embedding = self.photo_embedding_layer(browse_photo).to(self.device)
        browse_cor_query_embedding = self.corresponding_query_embedding(browse_photo).reshape(browse_photo.shape[0], browse_photo.shape[1], 768, query_num_per_photo).to(self.device)
        browse_cor_query_pinv = self.cor_query_pinv(browse_photo).reshape(browse_photo.shape[0], browse_photo.shape[1], query_num_per_photo, 768).to(self.device)
        browse_embedding = self.phi(browse_embedding)#batch, len, feature
        browse_embedding = self.iv(browse_embedding, browse_cor_query_embedding, browse_cor_query_pinv)#batch, len, feature
        browse_mask = torch.where(browse_photo==0, 1, 0).bool().to(self.device)
        browse_rep = self.browse_photo_att(browse_embedding, browse_mask)

        user_rep = self.user_rep(browse_rep) #batch,dense
        user_rep = user_rep.unsqueeze(dim=-1)#batch,dense,1

        logits = torch.matmul(photo_rep,user_rep)#batch,1,1
        prob = self.prob_sigmoid(logits).squeeze(-1).squeeze(-1)

        return prob





