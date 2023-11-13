import torch
import argparse
import os

from tqdm import tqdm
from model.architecture import MyArch
from utils.util import log_args
from model import modules



def load_MSC_data(data_path, args, idx, device, mode='test'):
    '''
    same format with data_loader part
    '''
    ### def __init__(self, data_path, device, args, mode='test'): 
        
    ### def __getitem__(self, idx): (same format with data_loader part)
    return 


def show_MSC_sample(model, args, device):
    data_path = '/home2/dataset/english_conversation/'
    dataset = os.listdir('/path/to/test_data')

    for idx in tqdm(range(len(dataset))):
        inputs = load_MSC_data(data_path, args, idx, device, mode='test')
        
        outputs = model.inference(inputs, greedy=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='MSC', help='select dataset for training')
    parser.add_argument('--act', default='relu', help='type of activation function')
    parser.add_argument('--max_length', type=int, default=60, help='maximum length of utterance')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint for load')
    args = parser.parse_args()
    log_args(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "/path/to/checkpoint"
    
    model = MyArch(args, device).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model.eval()
    
    show_MSC_sample(model, args, device)