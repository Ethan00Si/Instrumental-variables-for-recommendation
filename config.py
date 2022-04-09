import torch

train_batch_size = 512
test_batch_size = 30
class NRHUB_Config():
    def __init__(self):
        self.learning_rate = 0.0001
        self.epochs = 15
        self.interval = 100 #within each epoch, the interval of training steps to display loss
        self.save_epochs = 0
        self.save_path = './checkpoints/NRHUB/'
        self.load_path = './checkpoints/NRHUB/final.pth'
        self.device = 'cpu'
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.description = "NRHUB"

class IV_NRHUB_Config():
    def __init__(self):
        self.learning_rate = 0.0001
        self.epochs = 10
        self.interval = 100 
        self.save_epochs = 0
        self.save_path = './checkpoints/IV_nonlinear_mlp/'
        self.load_path = './checkpoints/IV_nonlinear_mlp/final.pth'
        self.device = 'cpu'
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.l2_lambda = 1e-4
        self.phi = {'hidden_dims':[768, 768],
                'dropout':[0.1],
                'is_dropout':True
        }
        self.alpha = {'hidden_dims':[768*2,512,128,32,1],
            'dropout':[],'is_dropout':False
        }
        self.beta = self.alpha
        self.description = 'IV4Rec_NRHUB'



class DIN_Config():
    def __init__(self):
        self.learning_rate = 0.0001
        self.epochs = 10
        self.interval = 100 
        self.save_epochs = 0
        self.load_path = './checkpoints/Din_baseline/final.pth'
        self.device = 'cpu'
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.description = "Din"

class IV_DIN_mlp_Config():
    def __init__(self):
        self.learning_rate = 1e-5
        self.epochs = 25
        self.interval = 1000 
        self.save_epochs = 0
        self.load_path = './checkpoints/IV_Din/final.pth'
        self.device = 'cpu'
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.l2_lambda = 1e-6
        self.phi = {'hidden_dims':[768,768,768],
                'dropout':[0.1,0.1],
                'is_dropout':True
        }
        self.alpha = {'hidden_dims':[768*2,512,128,32,1],
            'dropout':[],'is_dropout':False
        }
        self.beta = self.alpha
        self.description = 'IV_Din'
