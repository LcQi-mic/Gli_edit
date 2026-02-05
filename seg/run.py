import argparse

from trainer import Trainier


parser = argparse.ArgumentParser()

# Train
parser.add_argument("--optim_lr", default=5e-4, type=float, help="optimization learning rate")
parser.add_argument("--decay", default=0.1, type=float)
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epochs", default=150, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")

# Data
parser.add_argument("--data_json", default='/home/qlc/raid/dataset/Brats2023/ag/mri_gen.json', type=str, help="dataset json file")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")

# Model
parser.add_argument("--kernels", default=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], type=int)
parser.add_argument("--strides", default=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], type=list)
parser.add_argument("--n_class", default=3, type=list,  help="number classes of output image")
parser.add_argument("--img_ch", default=4, type=list, help="number channel of input image")

# Checkpoint
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint, including model, "
                                                       "optimizer, scheduler")

parser.add_argument("--logdir", default="/home/qlc/train_log", type=str, help="directory to save the tensorboard logs")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp

    trainer = Trainier(args)
    train_acc = trainer.train()
    
    return train_acc


if __name__ == "__main__":
    main()