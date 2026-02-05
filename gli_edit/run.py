import argparse
import numpy as np
import os
import torch
from trainer import Trainier


parser = argparse.ArgumentParser(description="")

# Train
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")

parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--rel_disc_loss", default=True, type=bool, help="Relativistic Discriminator Loss")
parser.add_argument("--device", default='cuda')

# Model
parser.add_argument("--image_size", default=256, type=int, help="Image size")
parser.add_argument("--style_dim", default=512, type=int, help="Style vector dimension")
parser.add_argument("--network_capacity", default=16, type=int, help="Network capacity")
parser.add_argument("--fmap_max", default=512, type=int, help="Max style vector dimension")
parser.add_argument("--cls", default=3, type=int, help="Output channel, gli is 3, mri is 1")
parser.add_argument("--style_depth", default=8, type=int, help="Style mapping network depth")
parser.add_argument("--task", default="gli_edit", type=str, help="gli_gen or gli_edit or mri_gen")
parser.add_argument("--attn_layers", default=[1,2,3,4], type=list, help="Attention layers")

# Text Encoder 
parser.add_argument("--width", default=512, type=int, help="transformer output dimension")
parser.add_argument("--layers", default=4, type=int, help="student model depth")
parser.add_argument("--heads", default=4, type=int, help="student model attention heads number")
parser.add_argument("--mlp_ratio", default=2.0, type=float)
parser.add_argument("--ls_init_value", default=None)
parser.add_argument("--batch_first", default=True, type=bool)
parser.add_argument("--teacher_model", default="ViT-B/16", type=str)

# Vision Encoder 
parser.add_argument("--embed_dim", default=24, type=int)
parser.add_argument("--window_size", default=[7,7], type=list)
parser.add_argument("--patch_size", default=[2,2], type=list)
parser.add_argument("--depths", default=[2,2,2,2,1], type=list)
parser.add_argument("--num_heads", default=[3,6,12,24,12], type=list)
parser.add_argument("--ve_mlp_ratio", default=4.0, type=float)
parser.add_argument("--qkv_bias", default=True, type=bool)
parser.add_argument("--drop_rate", default=0.0, type=float)
parser.add_argument("--attn_drop_rate", default=0.0, type=float)
parser.add_argument("--drop_path_rate", default=0.0, type=float)
parser.add_argument("--patch_norm", default=False, type=bool)
parser.add_argument("--out_c", default=1, type=int)
parser.add_argument("--in_c", default=1, type=int)
parser.add_argument("--mask_ratio", default=0.75, type=float)

# Losses
parser.add_argument("--l2_lambda", default=0.0, type=float, help="L2 loss weight for image generation")
parser.add_argument("--percep_lambda", default=1.0, type=float, help="Perceptural loss weight for all tasks")
parser.add_argument("--dice_lambda", default=1.0, type=float, help="Dice loss weight for glioma related tasks")
parser.add_argument("--adv_lambda", default=0.5, type=float, help="Adversarial loss weight for all tasks")

# Data
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
parser.add_argument("--data_json", default='/home/qlc/raid/dataset/Brats2023/ag/gli_edit.json', type=str, help="dataset json file")
parser.add_argument("--modality", default='t1c', type=str, help="Image modality")

# Checkpoint
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint, including model, "
                                                       "optimizer, scheduler")

parser.add_argument("--save_checkpoint", default=True, action="store_true", help="save checkpoint during training")

parser.add_argument("--gli_encoder", default=None, help="pre-trained gli encoder model")
parser.add_argument("--img_encoder", default=None, help="pre-trained mri encoder model")
parser.add_argument("--text_encoder", default=None, help="pre-trained text encoder model")
parser.add_argument("--clip", default="ViT-B/16", type=str)

parser.add_argument("--generator", default=None, help="pre-trained generator model for evaluation")

# Destination
parser.add_argument("--logdir", default="/home/qlc/train_log/gli_edit", type=str, help="directory to save the tensorboard logs")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    
    main_worker(args=args)


def main_worker(args):
    # Define gpus
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    trainer = Trainier(args=args)
    trainer.train()


if __name__ == "__main__":
    main()