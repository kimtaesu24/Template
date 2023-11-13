import torch
import argparse

from utils.util import set_random_seed, log_args
from trainer.trainer import MyTrainer


def run_mymodel(device, data_path, args):
    trainer = MyTrainer(device=device,
                        data_path=data_path,
                        )
    trainer.train_with_hyper_param(args=args)

def main(args):
    # Step 0. Initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(seed=args.seed, device=device)

    # Step 1. Load datasets
    if args.data_name == 'MSC':
        data_path = '/path/to/MSC'

    # Step 2. Run (train and evaluate) the specified model
    run_mymodel(device=device,
                data_path=data_path,
                args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_name', default='MSC', help='select dataset for training')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--act', default='relu', help='type of activation function')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--decay_rate', type=float, default=0.98, help='decay rate of learning rate')
    
    parser.add_argument('--save_at_every', type=int, default=5, help='save checkpoint')
    parser.add_argument('--metric_at_every', type=int, default=5, help='calculate metric scores')
    parser.add_argument('--resume', default=None, help='resume train with checkpoint path or not')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode for wandb')
    
    args = parser.parse_args()
    
    log_args(args)
    main(args)
