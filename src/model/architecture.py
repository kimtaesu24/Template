import torch
from torch import nn


class MyArch(torch.nn.Module):
    def __init__(
            self,
            args,
    ):
        super(MyArch, self).__init__()
        
        if args.act == 'relu':
            self.act = nn.ReLU()

    def forward(self, input, label, metric_log=False):
        '''
        input: list of inputs
        label: list of labels
        metric_log: whether calculate metric or not
        '''
        # return loss
        pass
        
    def inference(self, input):
        '''
        input: list of inputs
        '''
        # return output
        pass
