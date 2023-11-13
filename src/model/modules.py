import torch

def pad(inputs, padding_size):
    '''
    Used: dataset, model.inference()
    '''
    # tmp = [0 for i in range(max_length)]
    tmp = torch.zeros(padding_size)
    if len(inputs) > padding_size:
        tmp[:len(inputs)] = inputs[:padding_size]  # truncation
    else:
        tmp[:len(inputs)] = inputs  # padding
    return tmp
