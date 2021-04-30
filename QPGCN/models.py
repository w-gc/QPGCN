from layers import Layer, Fuse

import torch.nn as nn
import torch 
import torch.nn.functional as F


def fuse_outdim(in_dims, method = 'cat'):
    if method == 'cat':
        return sum(in_dims)
    elif method == 'avg' or method == 'mean' or method == 'sum':
        assert len(set(in_dims)) == 1, 'the dimensions of output MUST be SAME, or use `cat` method.'
        return in_dims[0]
    else:
        raise ValueError("The activation function '{}' has NOT been implemented yet ".format(method))


class GNN(nn.Sequential):
    def __init__(self, in_dim, adj, config):
        super(GNN, self).__init__()
        self.adj = adj
        self.len = len(config)
        self.in_dim = in_dim
        for i, cfg in enumerate(config):
            # 拼接层
            if cfg['name'] == 'Fuse':
                cfg["in_dim"] = sum([self[p].out_dim for p in cfg['input_path']])
                cfg["out_dim"] = fuse_outdim([self[p].out_dim for p in cfg['input_path']], method = cfg['fuse'])
                self.add_module(cfg['name'] + str(i), Fuse(cfg = cfg))
            else:
                if len(cfg['input_path']) == 0 or not 'input_path' in cfg:
                    cfg["in_dim"] = self.in_dim 
                else:
                    cfg["in_dim"] = self[cfg['input_path'][0]].out_dim
                self.add_module(cfg['name'] + str(i), Layer(adj = self.adj, cfg = cfg))
        self.out_dim = self[-1].out_dim
        
    def fixed_init(self, layer_num_list, data_dict_list):
        for layer_num, data_dict in zip(layer_num_list, data_dict_list):
            assert layer_num >= 0 or layer_num < len(self)
            assert isinstance(self[layer_num], Layer)
            self[layer_num].fixed_init(data_dict)

    def forward(self, x):
        output = []
        for layer in self:
            if isinstance(layer, Fuse):
                output.append(layer([output[p] for p in layer.input_path]))
            else:
                input = x if len(layer.input_path) == 0 else output[layer.input_path[0]]
                output.append(layer(input, self.adj))
        return output[-1]
    
    @property
    def get_support_learnable_param(self):
        out = [layer.get_support_learnable_param \
                for layer in self \
                if not isinstance(layer, Fuse) and layer.get_support_learnable_param is not None]
        return out

    def loss_spt_param(self, loss_fun = lambda x : x**2):
        loss = 0.0
        for layer in self:
            if isinstance(layer, Fuse):
                continue
            loss += layer.loss_spt_param(loss_fun=loss_fun)
        return loss

    def prediction(self, output):
        return torch.nn.functional.softmax(output, dim=1).argmax(1)
    
    def set_adj_cuda(self, device = None):
        if device is not None:
            self.adj = self.adj.cuda(device)
        else:
            self.adj = self.adj.cuda()