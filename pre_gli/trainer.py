import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler

from dataloader import get_loader
from models import GliCLIP
from vision_encoder import vision_encoder
from text_encoder import TextEncoder
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
        
        if self.args.amp:
            self.scaler = GradScaler()
            
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
            modality=self.args.modality,
            phase='train'
        )
    
        self.val_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            modality=self.args.modality,
            phase='val'
        )
        
        self.test_loader = get_loader(
            datalist_json=self.args.data_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            modality=self.args.modality,
            phase='test'
        )
        
        print('Train data number: {} | Val data number: {} | Test data number: {}'.format(
            len(self.train_loader) * self.args.batch_size,
            len(self.val_loader) * self.args.batch_size,
            len(self.test_loader) * self.args.batch_size))
        
    def init_model(self):                
        gli_encoder = vision_encoder(
                    in_chans=3,
                    embed_dim=self.args.embed_dim,
                    window_size=self.args.window_size,
                    patch_size=self.args.patch_size,
                    depths=self.args.depths,
                    num_heads=self.args.num_heads,
                    mlp_ratio=self.args.ve_mlp_ratio,
                    qkv_bias=self.args.qkv_bias,
                    drop_rate=self.args.drop_rate,
                    attn_drop_rate=self.args.attn_drop_rate,
                    drop_path_rate=self.args.drop_path_rate,
                    norm_layer=nn.LayerNorm,
                    patch_norm=self.args.patch_norm,
                    use_checkpoint=self.args.use_checkpoint
                )

        mri_encoder = vision_encoder(
                    in_chans=self.args.in_c,
                    embed_dim=self.args.embed_dim,
                    window_size=self.args.window_size,
                    patch_size=self.args.patch_size,
                    depths=self.args.depths,
                    num_heads=self.args.num_heads,
                    mlp_ratio=self.args.ve_mlp_ratio,
                    qkv_bias=self.args.qkv_bias,
                    drop_rate=self.args.drop_rate,
                    attn_drop_rate=self.args.attn_drop_rate,
                    drop_path_rate=self.args.drop_path_rate,
                    norm_layer=nn.LayerNorm,
                    patch_norm=self.args.patch_norm,
                    use_checkpoint=self.args.use_checkpoint
                )

        text_encoder = TextEncoder(
                    width=self.args.width,
                    layers=self.args.layers,
                    heads=self.args.heads,
                    mlp_ratio=self.args.mlp_ratio,
                    ls_init_value=self.args.ls_init_value,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    batch_first=True,
                    style_dim=self.args.style_dim
                )

        self.img_feature_dim = self.args.embed_dim * (2 ** len(self.args.depths))
        self.mri_encoder = self.load_model(mri_encoder, self.args.img_encoder_weights).to('cuda').eval()
        self.text_encoder = self.load_model(text_encoder, self.args.text_encoder_weights).to('cuda').eval()
        
        self.model = GliCLIP(
            style_dim=self.args.style_dim,
            img_feature_dim=self.img_feature_dim,
            gli_encoder=gli_encoder
        ).to('cuda')
                                
        pytorch_vision_encoder_params = sum(p.numel() for p in gli_encoder.parameters() if p.requires_grad)
        print("Vision and Gli encoder parameters count", pytorch_vision_encoder_params)
        
        pytorch_text_decoder_params = sum(p.numel() for p in text_encoder.parameters())
        print("Text encoder parameters count", pytorch_text_decoder_params)
        
    def config_optimizer(self):
        named_param = list(self.model.named_parameters())
        model_params = [p for n, p in named_param if p.requires_grad]
        
        self.optimizer = torch.optim.Adam(
            [
                {"params": model_params, "weight_decay": self.args.decay},
            ],
            lr=self.args.optim_lr
        )
        
        self.scheduler = None
        print("Config optimier")
        
    def config_losses_and_metrics(self):
        self.loss = Loss(batch_size=self.args.batch_size, 
                         temperature=self.args.temperature, 
                         alpha_1=self.args.alpha_1, 
                         alpha_2=self.args.alpha_2)
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

    def save_gli_encoder(self):
        state_dict = self.model.gli_encoder.state_dict() 
        save_dict = {"state_dict": state_dict}

        file_name = os.path.join(self.args.logdir, 'best_gli_encoder.pt')
        torch.save(save_dict, file_name)
        print("Saving checkpoint", file_name)
    
    def train_one_epoch(self):
        self.model.train()

        run_loss = AverageMeter()
        for idx, batch_data in enumerate(self.train_loader):
            image, gli, token = batch_data["image"].cuda(), batch_data["gli"].cuda(), batch_data["token"].cuda()

            gli = gli.to(torch.float32)
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                f_image = self.mri_encoder(image)
                f_text = self.text_encoder(token.to(torch.int32))    

            image_features, gli_features, text_features, logit_scale = self.model(f_image, gli, f_text)
            loss, _ = self.loss(image_features, gli_features, text_features, logit_scale)

            loss.backward()
            self.optimizer.step()

            run_loss.update(loss.item(), n=self.args.batch_size)

        return run_loss.avg
    
    def validata(self):
        self.model.eval()

        run_loss = AverageMeter()
        l2i_loss = AverageMeter()
        g2i_loss = AverageMeter()
        
        for idx, batch_data in enumerate(self.train_loader):
            image, gli, token = batch_data["image"].cuda(), batch_data["gli"].cuda(), batch_data["token"].cuda()
            gli = gli.to(torch.float32)
            with torch.no_grad():
                f_image = self.mri_encoder(image)
                f_text = self.text_encoder(token.to(torch.int32))    

                image_features, gli_features, text_features, logit_scale = self.model(f_image, gli, f_text)
                loss, (L2Iloss, G2Iloss) = self.loss(image_features, gli_features, text_features, logit_scale)

            run_loss.update(loss.item(), n=self.args.batch_size)
            l2i_loss.update(L2Iloss.item(), n=self.args.batch_size)
            g2i_loss.update(G2Iloss.item(), n=self.args.batch_size)
            
        return run_loss.avg, l2i_loss.avg, g2i_loss.avg
    
    def train(self):
        if self.trainer_initialized is False:
            self.initial_trainer()
            
            val_loss_min = 10.0
    
        for epoch in range(self.args.start_epoch, self.args.max_epochs):

            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss = self.train_one_epoch()

            print(
                "Train Epoch  {}/{}".format(epoch, self.args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            self.writer.add_scalar("train_loss", train_loss, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.args.val_every == 0:

                epoch_time = time.time()
                val_acc, L2Iloss, g2i_loss = self.validata()

                print(
                    "Validation stats {}/{}".format(epoch, self.args.max_epochs - 1),
                    ", val loss, L2Iloss:, G2Iloss:",
                    val_acc, L2Iloss, g2i_loss,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if self.writer is not None:
                    self.writer.add_scalar("Mean_Val_Loss", np.mean(val_acc), epoch)

                val_avg_acc = np.mean(val_acc)

                if val_avg_acc < val_loss_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_loss_min, val_avg_acc))
                    val_loss_min = val_avg_acc

                    self.save_checkpoint(file_name="ckpt_best.pt", epoch=epoch, best_acc=val_loss_min)
                    self.save_gli_encoder()
            self.save_checkpoint(file_name="ckpt_final.pt", epoch=epoch, best_acc=val_loss_min)

        print("Training Finished !, Best Accuracy: ", val_loss_min)
        return val_loss_min
    








