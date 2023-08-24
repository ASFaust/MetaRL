import torch

def count_parameters(dictionary):
    total_parameters = 0
    for key1, sub_dict in dictionary.items():
        for key2, tensor in sub_dict.items():
            total_parameters += torch.prod(torch.tensor(tensor.shape[1:])).item()
    return total_parameters

def flatten_params(params):
    ret_list = []
    for key1, sub_dict in params.items():
        for key2, tensor in sub_dict.items():
            ret_list.append(tensor.flatten())
    return torch.cat(ret_list)
