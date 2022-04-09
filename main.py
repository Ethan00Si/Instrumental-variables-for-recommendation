from models import IV_DIN, IV_NRHUB, NRHUB, DIN
from data.loader import get_dataloader
from config import *
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", default=None)
parser.add_argument("--eval", action="store_true", default=None)
parser.add_argument("--tune", action="store_true", default=None)
parser.add_argument("--load",action="store_true",default=None)
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--device", type=str, default="cpu")#cuda:1
parser.add_argument("--algos", type=str, default="NRHUB")
args = parser.parse_args()


model = None
config = None

if args.algos == "NRHUB":
    config = NRHUB_Config()
    config.device = args.device
    model = NRHUB.NRHUB(config)
elif args.algos == "IV4Rec_NRHUB":
    config = IV_NRHUB_Config()
    config.device = args.device
    model = IV_NRHUB.IV_nonlinear(config)
elif args.algos == "DIN":
    config = DIN_Config()
    config.device = args.device
    model = DIN.DeepInterestNetwork(config)
elif args.algos == "IV4Rec_DIN":
    config = IV_DIN_mlp_Config()
    config.device = args.device
    model = IV_DIN.DeepInterestNetwork(config)



if args.epochs > 0:
    config.epochs = args.epochs

if args.load:
    model._load_model(config.load_path)

model.to(device=config.device)

train_dataloader = get_dataloader(config.train_batch_size,'train')
validation_dataloader = get_dataloader(config.train_batch_size, 'validation')
test_dataloader = get_dataloader(config.test_batch_size,'test')

if args.train:
    model.fit(config, train_dataloader, tb=True)

if args.eval:
    metrics = model.evaluate(config, test_dataloader)
    print(metrics)

if args.tune:
    model.tune(config=config,loaders=[train_dataloader,test_dataloader],tb=True, is_save=True)

    # model._load_model(config.load_path) 
    # metrics = model.evaluate(config, test_dataloader)
    # print(metrics)

