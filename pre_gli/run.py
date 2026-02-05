import argparse

from trainer import Trainier
import os


parser = argparse.ArgumentParser()

# Train
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--decay", default=0.1, type=float)
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--alpha_1", default=0.5, type=float)
parser.add_argument("--alpha_2", default=0.5, type=float)
parser.add_argument("--temperature", default=0.05, type=float)

# Data
parser.add_argument("--data_json", default='/home/qlc/raid/dataset/Brats2023/ag/mri_gen.json', type=str, help="dataset json file")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--batch_size", default=64, type=int, help="number of batch size")
parser.add_argument("--modality", default='t1c', type=str, help="image modality")

# Text Encoder 
parser.add_argument("--width", default=512, type=int, help="transformer output dimension")
parser.add_argument("--style_dim", default=512, type=int, help="style vector dimension")
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
parser.add_argument("--use_checkpoint", default=False, type=bool)

# Checkpoint
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint, including model, "
                                                       "optimizer, scheduler")

parser.add_argument("--gli_encoder_weights", default=None)
parser.add_argument("--img_encoder_weights", default='/home/qlc/train_log/pre_img/t1c/best_mri_encoder.pt')
parser.add_argument("--text_encoder_weights", default='/home/qlc/train_log/pre_text/best_text_encoder.pt')

parser.add_argument("--logdir", default="/home/qlc/train_log/pre_gli", type=str, help="directory to save the tensorboard logs")


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
