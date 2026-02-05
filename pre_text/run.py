import argparse

from trainer import Trainier
import os


parser = argparse.ArgumentParser()

# Train
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")
parser.add_argument("--decay", default=0.1, type=float)
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--alpha_1", default=1., type=float)
parser.add_argument("--alpha_2", default=1., type=float)
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")

# Data
parser.add_argument("--data_json", default='/home/qlc/raid/dataset/Brats2023/ag/text_consis.json', type=str, help="dataset json file")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")

# Model
parser.add_argument("--width", default=512, type=int, help="transformer output dimension")
parser.add_argument("--style_dim", default=512, type=int, help="style vector dimension")
parser.add_argument("--layers", default=4, type=int, help="student model depth")
parser.add_argument("--heads", default=4, type=int, help="student model attention heads number")
parser.add_argument("--mlp_ratio", default=2.0, type=float)
parser.add_argument("--ls_init_value", default=None)
parser.add_argument("--batch_first", default=True, type=bool)
parser.add_argument("--teacher_model", default="ViT-B/16", type=str)

# Checkpoint
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint, including model, "
                                                       "optimizer, scheduler")

parser.add_argument("--transformer_weights", default=None, help="pre-trained transformer weights")
parser.add_argument("--logdir", default="/home/qlc/train_log/pre_text", type=str, help="directory to save the tensorboard logs")


def main():
    args = parser.parse_args()
    
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    trainer = Trainier(args)
    train_acc = trainer.train()
    
    return train_acc


if __name__ == "__main__":
    main()

