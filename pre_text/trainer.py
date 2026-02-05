import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler

from dataloader import get_loader
from models import DistillationModel
from loss import Loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
        
        
class Trainier():
    def __init__(self, args) -> None:
        self.args = args
            
        self.trainer_initialized = False
        
        self.start_epoch = self.args.start_epoch
            
    def initial_trainer(self):
        self.trainer_initialized = False
        self.config_dataset()
        self.init_model()
        self.config_optimizer()
        self.config_losses_and_metrics()
        self.config_wirter()
        
        self.trainer_initialized = True
        print("Trainer initialized.")
        
    def config_dataset(self):
        self.train_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            phase='train'
        )
    
        self.val_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            phase='val'
        )
        
        self.test_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            phase='test'
        )
        
        print('Train data number: {} | Val data number: {} | Test data number: {}'.format(
            len(self.train_loader) * self.args.batch_size,
            len(self.val_loader) * self.args.batch_size,
            len(self.test_loader) * self.args.batch_size))

    def init_model(self):
        self.model = DistillationModel(
            width=self.args.width,
            layers=self.args.layers,
            heads=self.args.heads,
            mlp_ratio=self.args.mlp_ratio,
            ls_init_value=self.args.ls_init_value,
            batch_first=self.args.batch_first,
            teacher_model=self.args.teacher_model,
            style_dim=self.args.style_dim
        ).to('cuda')
        
        self.model.s.initialize_parameters()
        print("Student model weights initialized")
        
        pytorch_encoder_params = sum(p.numel() for p in self.model.s.parameters() if p.requires_grad)
        print("Student model parameters count", pytorch_encoder_params)
        
    def config_optimizer(self):
        named_param = list(self.model.s.named_parameters())
        transformer_params = [p for n, p in named_param if p.requires_grad]
        
        self.optimizer = torch.optim.SGD(
            [
                {"params": transformer_params}
            ],
            lr=self.args.optim_lr, momentum=self.args.momentum
        )

        self.scheduler = None
        print("Config optimier")
        
    def config_losses_and_metrics(self):
        self.loss = Loss(self.args.alpha_1, self.args.alpha_2)
        print("Config losses and metrics")
        
    def config_wirter(self):
        self.writer = None
        if self.args.logdir is not None:
            self.writer = SummaryWriter(log_dir=f'{self.args.logdir}/tensorboard')
            print("Writing Tensorboard logs to ", f'{self.args.logdir}/tensorboard')
        print("Save model to ", self.args.logdir)

    def save_checkpoint(self, file_name, epoch, best_acc):
        state_dict = self.model.state_dict() 
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if self.optimizer is not None:
            save_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()

        file_name = os.path.join(self.args.logdir, file_name)
        torch.save(save_dict, file_name)
        print("Saving checkpoint", file_name)
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        self.model.load_state_dict(new_state_dict, strict=False)
        print("=> loaded model checkpoint")

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "optimizer" in checkpoint.keys():
            for k, v in checkpoint["optimizer"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            self.optimizer.load_state_dict(new_state_dict)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()      
            print("=> loaded optimizer checkpoint")
        if "scheduler" in checkpoint.keys():
            for k, v in checkpoint["scheduler"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            self.scheduler.load_state_dict(new_state_dict)
            self.scheduler.step(epoch=start_epoch)
            print("=> loaded scheduler checkpoint")
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(self.args.checkpoint, start_epoch, best_acc))
        
    def load_model(self, model, path):
        checkpoint = torch.load(path, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("=> loaded model checkpoint")
        return model
    
    def save_text_encoder(self):
        state_dict = self.model.s.state_dict() 
        save_dict = {"state_dict": state_dict}

        file_name = os.path.join(self.args.logdir, 'best_text_encoder.pt')
        torch.save(save_dict, file_name)
        print("Saving checkpoint", file_name)
    
    def train_one_epoch(self):
        self.model.s.train()
        self.model.t.eval()
        
        run_loss = AverageMeter()
        run_loss_dis = AverageMeter()
        run_loss_reg = AverageMeter()
        
        for idx, batch_data in enumerate(self.train_loader):
            token_0, token_1, token_2 = batch_data["token_0"], batch_data["token_1"], batch_data["token_2"]
            token_0, token_1, token_2 = token_0.cuda(), token_1.cuda(), token_2.cuda()
            
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                target_0 = self.model.get_t_feature(token_0)
                target_1 = self.model.get_t_feature(token_1)
                target_2 = self.model.get_t_feature(token_2)

            out_0 = self.model(token_0)
            out_1 = self.model(token_1)
            out_2 = self.model(token_2)

            loss, (loss_dis, loss_reg) = self.loss(out_0, out_1, out_2, target_0, target_1, target_2)

            loss.backward()
            self.optimizer.step()

            run_loss.update(loss.item(), n=self.args.batch_size)
            run_loss_dis.update(loss_dis.item(), n=self.args.batch_size)
            run_loss_reg.update(loss_reg.item(), n=self.args.batch_size)

        return run_loss.avg, run_loss_dis.avg, run_loss_reg.avg
    
    def validate(self):
        self.model.s.eval()
        self.model.t.eval()
        
        run_loss = AverageMeter()
        run_loss_dis = AverageMeter()
        run_loss_reg = AverageMeter()
        
        for idx, batch_data in enumerate(self.val_loader):
            token_0, token_1, token_2 = batch_data["token_0"], batch_data["token_1"], batch_data["token_2"]
            token_0, token_1, token_2 = token_0.cuda(), token_1.cuda(), token_2.cuda()
            
            with torch.no_grad():
                target_0 = self.model.get_t_feature(token_0)
                target_1 = self.model.get_t_feature(token_1)
                target_2 = self.model.get_t_feature(token_2)

                out_0 = self.model(token_0)
                out_1 = self.model(token_1)
                out_2 = self.model(token_2)

            loss, (loss_dis, loss_reg) =  self.loss(out_0, out_1, out_2, target_0, target_1, target_2)

            run_loss.update(loss.item(), n=self.args.batch_size)
            run_loss_dis.update(loss_dis.item(), n=self.args.batch_size)
            run_loss_reg.update(loss_reg.item(), n=self.args.batch_size)

        return run_loss.avg, run_loss_dis.avg, run_loss_reg.avg
    
    def train(self):
        if self.trainer_initialized is False:
            self.initial_trainer()
            
        val_loss_min = 10.0
    
        for epoch in range(self.args.start_epoch, self.args.max_epochs):

            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss, distill_loss, reg_loss = self.train_one_epoch()

            print(
                "Train Epoch  {}/{}".format(epoch, self.args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "distill_loss: {:.4f}".format(distill_loss),
                "reg_loss: {:.4f}".format(reg_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            self.writer.add_scalar("train_loss", train_loss, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.args.val_every == 0:

                epoch_time = time.time()
                val_loss, distill_loss, reg_loss = self.validate()

                print(
                    "Validation stats {}/{}".format(epoch, self.args.max_epochs - 1),
                    ", loss: {:.4f}".format(val_loss),
                    "distill_loss: {:.4f}".format(distill_loss),
                    "reg_loss: {:.4f}".format(reg_loss),
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if self.writer is not None:
                    self.writer.add_scalar("Mean_Val_Loss", np.mean(val_loss), epoch)

                val_avg_acc = val_loss

                if val_avg_acc < val_loss_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_loss_min, val_avg_acc))
                    val_loss_min = val_avg_acc

                    self.save_checkpoint(file_name="ckpt_best.pt", epoch=epoch, best_acc=val_loss_min)
                    self.save_text_encoder()
                    
                self.save_checkpoint(file_name="ckpt_final.pt", epoch=epoch, best_acc=val_loss_min)

        print("Training Finished !, Best Accuracy: ", val_loss_min)
        return val_loss_min
    







