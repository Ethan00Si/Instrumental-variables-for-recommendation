import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.metric import cal_metric_mind

import logging
import time
import datetime
# import torchsnooper

from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()

# log
log_file_name = "./log/"+str(datetime.datetime.now().year)+'_'\
    +str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'.log'
logging.basicConfig(format='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG, filename=log_file_name, filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s'))
logging.getLogger(__name__).addHandler(console)


class BasicModel(nn.Module):
    def __init__(self, config):
        """
        basic model:
            create model and optimizer
            initialize model and hyper-parameter
        """
        super().__init__()
        self.device = config.device
        self.name = 'basic'
        self.regularization_weight = []


    def _save_model(self, info, model_save_path):
        """
            save model
        """
        logging.info('Saving to %s%s.pth' % (model_save_path, info))
        torch.save(self.state_dict(), '%s%s.pth' % (model_save_path, info))


    def _load_model(self, load_path):
        """
            loading model
        """
        logging.info('Loading from %s' % (load_path))
        self.load_state_dict(torch.load(load_path, map_location=self.device))
        # self.load_state_dict(torch.load(load_path))

    def _log(self, res, config=None, cur_epoch=None):
        """
            wrap logging
        """
        logging.info("evaluation results:\n {}".format(res))
        import os
        save_directory = "./log/{}".format(self.name)
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        with open("./log/{}/{}_performance.log".format(self.name, config.description), "a+") as f:
            f.write(self.name+"\n")
            if config is not None:
                f.write("config:\n")
                f.write(str(config.__dict__)+"\n")
                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            if cur_epoch is not None:
                f.write("current epoch: {}\n".format(str(cur_epoch)))
            f.write("metrics:\n")
            f.write(str(res)+"\n")

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))


    def get_regularization_loss(self, ):
        '''
            calculate regularization loss(l1 or l2)
        '''
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def _run_eval(self, dataloader):
        """ making prediction and gather results into groups according to impression_id
        """
        all_imp_pred = {}
        all_imp_label = {}
        
        for batch_data in tqdm(iterable=dataloader, mininterval=1, ncols=120):
            batch_data_input, batch_label_input , batch_impression= batch_data[:2],batch_data[2],batch_data[-1]
            # batch_label_input = batch_label_input.to(self.device)
            pred = self(batch_data_input).squeeze(-1).tolist()
            label = batch_label_input.squeeze(-1).tolist()
            # k += 1
            # if k>4:
            #     break
            for i, imp in enumerate(batch_impression):
                if  all_imp_pred.__contains__(imp.item()) == False:
                    all_imp_pred[imp.item()] = []
                    all_imp_label[imp.item()] = []
                all_imp_pred[imp.item()].append(pred[i])
                all_imp_label[imp.item()].append(label[i])
            

        return all_imp_pred, all_imp_label

    def evaluate(self, config, dataloader, log=True):
        """
        Evaluate the given file and returns some evaluation metrics.
        Args:
            self(nn.Module)
            config(dict)
            dataloader(torch.utils.data.DataLoader)
            loading(bool): whether to load model
            log(bool): whether to log
        Returns:
            res(dict): A dictionary contains evaluation metrics.
        """
        
        self.eval()

        logging.info("evaluating...")

        preds, labels = self._run_eval(dataloader)

        # calculate metrics
        res = cal_metric_mind(preds, labels)


        self.train()

        return res

    # @torchsnooper.snoop()
    def _run_train(self, dataloader, optimizer, loss_func, epochs, writer=None, interval=100, save_epochs=0, is_save=False):
        """ train model and print loss meanwhile
        """
        total_loss = 0
        total_steps = 0

        for epoch in range(epochs):
            epoch_loss = 0
            tqdm_ = tqdm(iterable=dataloader, mininterval=1, ncols=120)
            for step, x in enumerate(tqdm_):

                optimizer.zero_grad()
                data, label = x[:-1],x[-1]
                label = label.to(self.device)
                pred = self(data)

                loss = loss_func(pred, label)
                reg_loss = self.get_regularization_loss()
                all_loss = loss + reg_loss

                epoch_loss += all_loss.item()
                total_loss += all_loss.item()

                total_steps += 1

                all_loss.backward()
                optimizer.step()

                if step % interval == 0 and step > 0:
                    tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                    if writer:
                        # for name, param in self.named_parameters():
                        #     writer.add_histogram(name, param, step)

                        writer.add_scalar("training_loss",
                                        total_loss/total_steps, step+epoch*dataloader.__len__())

            if save_epochs > 0:
                if epoch % save_epochs == 0 and epoch > 0:
                    self._save_model(epoch,  "./checkpoints/{}/".format(self.name))

                

            if writer:
                writer.add_scalar("epoch_loss", epoch_loss/dataloader.__len__(), epoch)
                for name, param in self.named_parameters():
                            writer.add_histogram(name, param, epoch)


        if is_save:
            self._save_model("final",  "./checkpoints/{}/".format(self.name))

    def fit(self, config, loaders, tb=False):
        """ wrap training process
        Args:
            model(torch.nn.Module): the model to be trained
            loaders: torch.utils.data.DataLoader
            config(class object): hyper paramaters
            en: shell parameter
        """
        self.train()
        writer = None

        if tb:
            writer = SummaryWriter("log/tb/{}/{}/{}/".format(
                self.name, config.description, datetime.datetime.now().strftime("%Y%m%d-%H")))

        # in case the folder does not exists, create one
        import os
        save_directory = "./checkpoints/{}".format(self.name)
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)


        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        self._run_train(loaders, optimizer, loss_func,
                        config.epochs, 
                        save_path=config.save_path,
                        # writer=writer, 
                        interval=config.interval,
                        save_epochs=config.save_epochs,
                        config = config
                        )

    def tune_evaluate(self, config, dataloader, loss_func, eval_loss, cur_steps, writer, log=True):
        """
        Evaluate the given file and returns some evaluation metrics.
        Args:
            self(nn.Module)
            config(dict)
            dataloader(torch.utils.data.DataLoader)
            loss_func(torch.nn.loss_function)
            eval_loss(float):loss on validation set(test set)
            cur_steps(int):total steps 
            log(bool): whether to log
        Returns:
            res(dict): A dictionary contains evaluation metrics.
        """
        
        self.eval()

        logging.info("evaluating...")

        preds, labels = self._run_eval(dataloader)

        res = cal_metric_mind(preds, labels)

        print('auc: %s\n'%res['auc'])

        if log:
            res["epoch"] = config.epochs
            res["learning_rate"] = config.learning_rate
            res["train_batch_size"] = config.train_batch_size
            res["test_batch_size"] = config.test_batch_size
            self._log(res, config)

        self.train()

        return res, eval_loss, cur_steps

    def _run_tune(self, loaders, optimizer, loss_func, config,  writer=None, interval=100, save_epochs=0, is_save=False):
        """ train model and evaluate on validation set once every epoch.
            Early stop after 5 epochs without improvement.
        """
        total_loss = 0
        total_steps = 0

        best_res = {"auc":0}
        early_stops = 0

        eval_steps = 0
        eval_loss = 0
        for epoch in range(config.epochs):
            epoch_loss = 0
            tqdm_ = tqdm(iterable=loaders[0], mininterval=1, ncols=120)
            for step, x in enumerate(tqdm_):
                
                optimizer.zero_grad()

                data, label = x[:-1],x[-1]
                label = label.to(self.device)
                pred = self(data)
                
                loss = loss_func(pred, label)
                reg_loss = self.get_regularization_loss()
                all_loss = loss + reg_loss

                epoch_loss += all_loss.item()
                total_loss += all_loss.item()
                total_steps += 1

                all_loss.backward()
                optimizer.step()


                if step % interval == 0 and step > 0:

                    tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, epoch_loss / step))
                    if writer:
                        # for name, param in self.named_parameters():
                        #     writer.add_histogram(name, param, step)

                        writer.add_scalar("training_loss",
                                        total_loss/total_steps, step+epoch*loaders[0].__len__())

            with torch.no_grad():
                result, eval_loss, eval_steps = self.tune_evaluate(config, 
                    loaders[1], loss_func,  eval_loss, eval_steps, writer, log=False)

                if writer:
                    for k,v in result.items():
                        writer.add_scalar("metric/"+str(k), v, epoch)
                result["epoch"] = epoch+1
                result["step"] = step

                logging.info("current result of {} is {}".format(self.name, result))
                self._log(result, config=config,cur_epoch=epoch)
                if result["auc"] > best_res["auc"]:
                    best_res = result
                    logging.info("current metrics: ", best_res)
                    if is_save:
                        self._save_model("best_"+config.description,  "./checkpoints/{}/".format(self.name))

                elif result["auc"] - best_res["auc"] <= 0.0:
                    early_stops += 1
                    if early_stops >= 5:
                        print('overfitting! Early stop!')
                        logging.info("model is overfitting, the result is {}, force shutdown".format(result))
                        return best_res

            if save_epochs > 0:
                if epoch % save_epochs == 0 and epoch > 0:
                    self._save_model(epoch,  "./checkpoints/{}/".format(self.name))

                

            if writer:
                # writer.add_scalar("epoch_loss", epoch_loss/loaders[0].__len__(), epoch)
                for name, param in self.named_parameters():
                            writer.add_histogram(name, param, epoch)

        return best_res  

    def tune(self, config, loaders, tb=False, is_save=False):
        """ train and evaluate sequentially
        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            config(dict): hyper paramaters
            en: shell parameter
        """

        self.train()
        writer = None

        # in case the folder does not exists, create one
        import os
        save_directory = "./checkpoints/{}".format(self.name)
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        if tb:
            writer = SummaryWriter("log/tb/{}/{}/{}/".format(
                self.name, config.description, datetime.datetime.now().strftime("%Y%m%d-%H")))


        logging.info("training...")
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        

        res = self._run_tune(loaders, optimizer, loss_func, config, 
                        writer=writer, interval=config.interval, save_epochs=config.save_epochs, is_save=is_save)

        self._log(res, config)
