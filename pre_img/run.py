import argparse

from trainer import Trainier
import os

parser = argparse.ArgumentParser()

# Train
parser.add_argument("--optim_lr", default=5e-4, type=float, help="optimization learning rate")
parser.add_argument("--decay", default=0.1, type=float)
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")

# Data
parser.add_argument("--data_json", default='/home/qlc/raid/dataset/Brats2023/ag/mri_gen.json', type=str, help="dataset json file")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
parser.add_argument("--modality", default='t1c', type=str, help="image modality")

# Model
parser.add_argument("--embed_dim", default=24, type=int)
parser.add_argument("--window_size", default=[7,7], type=list)
parser.add_argument("--patch_size", default=[2,2], type=list)
parser.add_argument("--depths", default=[2,2,2,2,1], type=list)
parser.add_argument("--num_heads", default=[3,6,12,24,12], type=list)
parser.add_argument("--mlp_ratio", default=4.0, type=float)
parser.add_argument("--qkv_bias", default=True, type=bool)
parser.add_argument("--drop_rate", default=0.0, type=float)
parser.add_argument("--attn_drop_rate", default=0.0, type=float)
parser.add_argument("--drop_path_rate", default=0.0, type=float)
parser.add_argument("--patch_norm", default=False, type=bool)
parser.add_argument("--out_c", default=1, type=int)
parser.add_argument("--in_c", default=1, type=int)
parser.add_argument("--mask_ratio", default=0.75, type=float)
parser.add_argument("--use_checkpoint", default=False, type=bool)

# Checkpoint
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint, including model, "
                                                       "optimizer, scheduler")

parser.add_argument("--encoder", default=None, help="pre-trained encoder weights")
parser.add_argument("--logdir", default="/home/qlc/train_log", type=str, help="directory to save the tensorboard logs")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    trainer = Trainier(args)
    train_acc = trainer.train()
    
    return train_acc


if __name__ == "__main__":
    main()