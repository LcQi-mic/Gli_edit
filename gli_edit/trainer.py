import os

import torch
from torch import autograd  
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import grad as torch_grad

from models import Generator, Discriminator
from loss import Perceptual_loss, WeightedDiceLoss
from dataloader import get_loader

import time
import clip
from vision_encoder import vision_encoder
from text_encoder import TextEncoder
from monai.metrics import DiceMetric


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
        

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class Trainier:
    def __init__(self, 
                args=None):
        self.args = args
        self.global_step = 0
        
        self.rel_disc_loss = self.args.rel_disc_loss
        self.trainer_initialized = False
        self.device = self.args.device
        

    def initial_trainer(self):
        self.trainer_initialized = False
        self.config_dataset()
        self.init_model()
        self.config_optimizer()
        self.config_loss()
        self.config_wirter()
        self.trainer_initialized = True
        
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
        self.gli_encoder = None
        self.text_encoder = None
        
        if self.args.text_encoder is not None:      
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
            self.text_encoder = self.load_encoder(text_encoder, self.args.text_encoder)
            self.text_encoder.to(self.device).eval()

        if self.args.gli_encoder is not None:
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
                    use_checkpoint=False
                )
            self.gli_encoder = self.load_encoder(gli_encoder, self.args.gli_encoder)
            self.gli_encoder.to(self.device).eval()

        self.image_feature_dim = self.args.embed_dim * (2 ** len(self.args.depths))
        self.G = Generator(    
            image_size=self.args.image_size,
            style_dim=self.args.style_dim,
            network_capacity=self.args.network_capacity,
            fmap_max=self.args.fmap_max,
            cls=self.args.cls,
            style_depth=self.args.style_depth,
            task=self.args.task,
            attn_layers=self.args.attn_layers,
            image_feature_dim=self.image_feature_dim).to(self.device)
        
        self.G._init_weights()
        print("G weight initialized.")
        
        self.D = Discriminator(
            image_size=self.args.image_size,
            network_capacity=self.args.network_capacity,
            attn_layers=self.args.attn_layers,        
            fmap_max=self.args.fmap_max,  
            cls=self.args.cls).to(self.device)

        self.sigmoid = nn.Sigmoid()
        
        pytorch_encoder_params = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
        print("G parameters count", pytorch_encoder_params)
        
        pytorch_decoder_params = sum(p.numel() for p in self.D.parameters() if p.requires_grad)
        print("D parameters count", pytorch_decoder_params)
        
    def config_optimizer(self):
        self.G_optimizer = torch.optim.Adam(list(self.G.parameters()), 
                                            lr=self.args.optim_lr,
                                            betas=(0.5, 0.9))
        self.apply_gradient_penalty = True
        
        self.D_optimizer = torch.optim.Adam(list(self.D.parameters()),
                                            lr=self.args.optim_lr,
                                            betas=(0.5, 0.9))
        self.scheduler  = None
        
    def config_loss(self):
        self.dice_loss = WeightedDiceLoss()
        self.perceptual_loss = Perceptual_loss(self.gli_encoder, 3)
        self.dice = DiceMetric(include_background=True, reduction='mean', num_classes=self.args.cls)
        
    def config_wirter(self):
        self.writer = None
        if self.args.logdir is not None:
            self.writer = SummaryWriter(log_dir=f'{self.args.logdir}/tensorboard')
            print("Writing Tensorboard logs to ", f'{self.args.logdir}/tensorboard')
        print("Save model to ", self.args.logdir)
        
    def train_one_epoch(self):
        self.G.train()
        self.D.train()
        run_g_loss = AverageMeter()
        run_dice_loss = AverageMeter()
        run_d_loss = AverageMeter()
        run_lpips_loss = AverageMeter()       
        
        for idx, batch_data in enumerate(self.train_loader):
            gli_0, token_0 = batch_data['gli_0'].to('cuda'), batch_data['token_0'].to('cuda')
            gli_1, token_1 = batch_data['gli_1'].to('cuda'), batch_data['token_1'].to('cuda')
            gli_0 = gli_0.to(torch.float32)
            gli_1 = gli_1.to(torch.float32)
            with torch.no_grad():
                if self.text_encoder is not None:
                    f_text = self.text_encoder(token_1.to(torch.int32))
                else:
                    f_text=None

                if self.gli_encoder is not None:
                    f_gli = self.gli_encoder(gli_0)
                else: 
                    f_gli = None

            self.G_optimizer.zero_grad()
            
            w = self.G.get_w(gli_0, f_text=f_text, f_gli=f_gli)
            input_noise = self.G.make_image_noise(gli_0)
            y_hat = self.G(w, input_noise, f_gli)
            
            # adversarial loss
            if self.args.adv_lambda > 0: 
                d_loss_dict = self.train_discriminator(real_img=gli_1, fake_img=y_hat)

            loss, loss_dict = self.calc_loss(y=gli_1, y_hat=y_hat)
            
            loss.backward()
            self.G_optimizer.step()

            run_g_loss.update(loss_dict['loss_g'], n=self.args.batch_size)
            run_dice_loss.update(loss_dict['loss_dice'], n=self.args.batch_size)
            run_d_loss.update(d_loss_dict['loss_d'], n=self.args.batch_size)
            run_lpips_loss.update(loss_dict['loss_lpips'], n=self.args.batch_size)            

        return run_g_loss.avg, run_dice_loss.avg, run_d_loss.avg, run_lpips_loss.avg

    def validate(self):
        self.G.eval()
        
        g_loss = AverageMeter()
        dice_loss = AverageMeter()
        d_loss = AverageMeter()
        dice = AverageMeter()
        
        for idx, batch_data in enumerate(self.train_loader):
            gli_0, token_0 = batch_data['gli_0'].to('cuda'), batch_data['token_0'].to('cuda')
            gli_1, token_1 = batch_data['gli_1'].to('cuda'), batch_data['token_1'].to('cuda')
            gli_0 = gli_0.to(torch.float32)
            gli_1 = gli_1.to(torch.float32)
            with torch.no_grad():
                if self.text_encoder is not None:
                    f_text = self.text_encoder(token_1.to(torch.int32))
                else:
                    f_text=None

                if self.gli_enocder is not None:
                    f_gli = self.gli_enocder(gli_0)
                else: 
                    f_gli = None
                    
                w = self.G.get_w(gli_1, f_text=f_text, f_gli=f_gli)
                input_noise = self.G.make_image_noise(gli_1)
                y_hat = self.G(w, input_noise, f_gli)
                
                y_hat = self.sigmoid(y_hat)
                y_hat[y_hat > 0.5] = 1.
                
                d_loss_dict = self.validate_discriminator(real_img=gli_1, fake_img=y_hat)
                loss, loss_dict = self.calc_loss(y=gli_1, y_hat=y_hat)
                dice_metric = self.dice(y_hat, gli)
                
            g_loss.update(loss_dict['loss_g'], n=self.args.batch_size)
            dice_loss.update(loss_dict['loss_dice'], n=self.args.batch_size)
            d_loss.update(d_loss_dict['loss_d'], n=self.args.batch_size)
            dice.update(dice_metric, n=self.args.batch_size )

        self.G.train()
        return g_loss.avg, dice_loss.avg, d_loss.avg, dice.avg
        
    def train(self):
        if self.trainer_initialized is False:
            self.initial_trainer()
            
        G_loss_min = 10.0
    
        for epoch in range(self.args.start_epoch, self.args.max_epochs):

            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            run_g_loss, run_dice_loss, run_d_loss, run_loss_lpips = self.train_one_epoch()

            print(
                "Train Epoch  {}/{}".format(epoch, self.args.max_epochs - 1),
                "g_loss: {:.4f}".format(run_g_loss),
                "dice_loss: {:.4f}".format(run_dice_loss),
                "d_loss: {:.4f}".format(run_d_loss),
                "loss_lpips: {:.4f}".format(run_loss_lpips),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            self.writer.add_scalar("g_loss", run_g_loss, epoch)
            self.writer.add_scalar("dice_loss", run_dice_loss, epoch)
            self.writer.add_scalar("d_loss", run_d_loss, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.args.val_every == 0:

                epoch_time = time.time()
                val_g_loss, val_dice_loss, val_d_loss, dice = self.validate()

                print(
                    "Final validation stats {}/{}".format(epoch, self.args.max_epochs - 1),
                    ", val_g_loss | val_dice_loss | val_d_loss | dice:",
                    val_g_loss, val_dice_loss, val_d_loss, dice,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if self.writer is not None:
                    self.writer.add_scalar("Mean_Val_G_Loss", np.mean(val_g_loss), epoch)

                val_dice_loss = np.mean(dice)

                if val_dice_loss < G_loss_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(G_loss_min, val_dice_loss))
                    G_loss_min = val_dice_loss

                    self.save_checkpoint(file_name="ckpt_best.pt", epoch=epoch, best_acc=G_loss_min)
                    self.save_G(file_name="gli_gen.pt")

                self.save_checkpoint(file_name="ckpt_final.pt", epoch=epoch, best_acc=G_loss_min)

        print("Training Finished !, Best Accuracy: ", G_loss_min)
        return G_loss_min

    def save_checkpoint(self, file_name, epoch, best_acc):
        state_dict = self.G.state_dict() 
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if self.G_optimizer is not None:
            save_dict["optimizer"] = self.G_optimizer.state_dict()
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

    def load_encoder(self, model, path):
        checkpoint = torch.load(path, map_location="cpu")
        from collections import OrderedDict
        
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("=> loaded model checkpoint")
        return model
    
    def save_G(self):
        state_dict = self.G.state_dict() 
        save_dict = {"state_dict": state_dict}

        file_name = os.path.join(self.args.logdir, 'best_gli_encoder.pt')
        torch.save(save_dict, file_name)
        print("Saving checkpoint", file_name)
        
    def load_G(self):
        G = torch.load(self.args.generator, map_location=torch.device('cpu'))
        return G

    def calc_loss(self, y, y_hat):
        loss_dict = {}
        loss = 0.0
        
        if self.args.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.args.l2_lambda
        if self.args.percep_lambda > 0:
            loss_lpips = self.perceptual_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.args.percep_lambda
        if self.args.dice_lambda > 0:
            loss_dice = self.dice_loss(y_hat, y)
            loss_dict['loss_dice'] = float(loss_dice)
            loss += loss_dice * self.args.dice_lambda
        if self.args.adv_lambda > 0:  ##### modified
            y_hat = self.sigmoid(y_hat)
            y_hat[y_hat > 0.5] = 1.
            loss_g = F.softplus(-self.D(y_hat).detach()).mean()
            loss_dict['loss_g'] = float(loss_g)
            loss += loss_g * self.args.adv_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict
    
    ##### modified
    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['loss_d_real'] = float(real_loss)
        loss_dict['loss_d_fake'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, real_img, fake_img):
        loss_dict = {}
        self.requires_grad(self.D, True)
        
        fake_img = self.sigmoid(fake_img)
        fake_img[fake_img > 0.5] = 1.

        real_pred = self.D(real_img.to(torch.float32))
        fake_pred = self.D(fake_img.detach())
        
        if self.rel_disc_loss:
            real_output_loss = real_pred - fake_pred.mean()
            fake_output_loss = fake_pred - real_pred.mean()
        
        loss = self.discriminator_loss(real_output_loss, fake_output_loss, loss_dict)
        loss_dict['loss_d'] = float(loss)
        loss = loss * self.args.adv_lambda 
        
        # if self.apply_gradient_penalty:
        #     gp = gradient_penalty(real_img.requires_grad_(), real_pred)
        #     self.last_gp_loss = gp.clone().detach().item()
        #     self.track(self.last_gp_loss, 'GP')
        #     disc_loss = disc_loss + gp

        self.D_optimizer.zero_grad()
        loss.backward()
        self.D_optimizer.step()
    
        # Reset to previous state
        self.requires_grad(self.D, False)

        return loss_dict
    
    def validate_discriminator(self, real_img, fake_img):
        with torch.no_grad():
            loss_dict = {}
            real_pred = self.D(real_img.to(torch.float32))
            fake_pred = self.D(fake_img.detach())
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['loss_d'] = float(loss)
            loss = loss * self.args.adv_lambda 
            return loss_dict