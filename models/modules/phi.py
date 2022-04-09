import torchsnooper
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "MLP"

        self.hidden_dims = config['hidden_dims']
        self.droput = config['dropout']
        self.is_dropout = config['is_dropout']

        self.activation = nn.Tanh()

        for i in range(1,len(self.hidden_dims)):
            setattr(self, "linear_"+str(i),
                    nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            # setattr(self, 'batchNorm_'+str(i),
            #         nn.BatchNorm1d(self.hidden_dims[i]))
            if config["is_dropout"]:
                setattr(self, 'dropout_'+str(i),
                        nn.Dropout(self.droput[i-1]))

        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(1,len(self.hidden_dims)):
            nn.init.xavier_uniform_(getattr(self,"linear_"+str(i)).weight)

    # @torchsnooper.snoop()
    def forward(self, x):
        #batch, len, feature
        deep_out = x
        for i in range(1, len(self.hidden_dims)):
            deep_out = getattr(self, 'linear_'+str(i))(deep_out)
            deep_out = self.activation(deep_out)
            if self.is_dropout:
                deep_out = getattr(self, "dropout_"+str(i))(deep_out)
        return deep_out
