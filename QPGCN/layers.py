import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules import Dropout
from torch.nn import Sequential
from support_utils import pre_support, Identity
from copy import deepcopy

def activation(name):
    if name == 'none':
        act = lambda x: x
    elif name == 'relu':
        from torch.nn import ReLU
        act = ReLU()
    elif name == 'elu':
        from torch.nn import ELU
        act = ELU()
    elif name == 'softmax':
        from torch.nn import Softmax
        act = Softmax(dim=1)
    else:
        raise ValueError("The activation function '{}' has NOT been implemented yet ".format(name))
    return act


def fuse_features(input, method = 'cat'):
    if method == 'cat':
        return torch.cat(input, dim = 1)
    elif method == 'avg' or method == 'mean':
        return sum(input) / len(input)
    elif method == 'sum':
        return sum(input)
    else:
        raise ValueError("The activation function '{}' has NOT been implemented yet ".format(method))


class Fuse(Module):
    def __init__(self, cfg):
        super(Fuse, self).__init__()
        self.in_dim = cfg['in_dim']
        self.out_dim = cfg['out_dim']
        self.act = activation(cfg['activation'])
        self.drop = Dropout(cfg['dropout'])
        self.input_path = cfg['input_path']
        self.fuse = cfg['fuse']

        if cfg['bias']:
            self.bias = Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        x = fuse_features(input, method=self.fuse)
        x = self.drop(x)
        if self.act is not None:
            x = self.act(x)
        
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'


class Layer(Module):
    r"""
        :math:`X_out = \sigma ( S X_int \Theta + B )`
    """
    def __init__(self, adj, cfg):
        super(Layer, self).__init__()
        self.in_dim = cfg['in_dim']
        self.out_dim = cfg['out_dim']
        self.act = activation(cfg['activation'])
        self.drop = Dropout(cfg['dropout'])
        self.norm = PairNorm(mode = cfg['PairNorm']['mode'] if 'PairNorm' in cfg else 'None', 
                             scale = cfg['PairNorm']['scale'] if 'PairNorm' in cfg else 1.0)

        self.input_path = cfg['input_path']

        self.support_learnable = 'learnable' in  cfg['support']
        self.support_dict = deepcopy(cfg['support'])

        assert len(self.input_path) <= 1, 'the length of the `input_path` of the Layer\'s object MUST be 0 or 1.'

        self.weight = Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        if cfg['bias']:
            self.bias = Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)

        if self.support_learnable:
            self.init_support_param()
            if self.support_dict['name'] == 'DeltaQPF':
                self.Adj_tilde = pre_support(adj, sup_dict={'name': 'Adj_tilde', 'gamma': self.support_dict['gamma']}, sparse=True)
                self.support_dict['C'] = - 1
                self.support_dict['degrees'] = torch.sparse.sum(adj, dim = 1).to_dense()
        else:
            self.support = pre_support(adj, self.support_dict)

        self.init_parameters()

    def fixed_init(self, data_dict):
        for param_name in data_dict.keys():
            self._parameters[param_name].data = data_dict[param_name]
    
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def init_support_param(self):
        if not self.support_learnable:
            return None
        for param_name in self.support_dict['learnable']:
            self.register_parameter(param_name, Parameter(torch.rand(1)))
            if len(self.support_dict[param_name]) == 0:
                continue
            elif len(self.support_dict[param_name]) == 1:
                self._parameters[param_name].data = torch.FloatTensor(self.support_dict[param_name])
            elif len(self.support_dict[param_name]) == 2:
                self._parameters[param_name].data.uniform_(self.support_dict[param_name][0], \
                                                        self.support_dict[param_name][1])
            else:
                raise ValueError("The length of the init value of support parameter MUST be 0, 1 or 2.")
                
    @property
    def get_support_learnable_param(self):
        if self.support_learnable:
            return [self._parameters[param].data.item() for param in self.support_dict['learnable']]
    
    def loss_spt_param(self, loss_fun = lambda x : x**2):
        loss = 0.0
        if self.support_learnable:
            for param in self.support_dict['learnable']:
                loss += loss_fun(self._parameters[param])
        return loss

    def forward(self, input, adj):
        x = self.drop(input)
        x = torch.mm(x, self.weight)

        if self.support_learnable:
            for param in self.support_dict['learnable']:
                self.support_dict[param] = self._parameters[param]

            if self.support_dict['name'] == 'DeltaQPF':
                self.Adj_tilde = self.Adj_tilde.to(x.device)
                self.support_dict['degrees'] = self.support_dict['degrees'].to(x.device)
                self.support = pre_support(self.Adj_tilde, self.support_dict, sparse=False)
                x = torch.mm(self.support.to(x.device), x) + torch.mm(self.Adj_tilde, torch.mm(self.Adj_tilde, -x))
                
                for _ in range(self.support_dict['power'] - 1) :
                    x = torch.mm(self.support.to(x.device), x) + torch.mm(self.Adj_tilde, torch.mm(self.Adj_tilde, -x))
            else:
                self.support = pre_support(adj, self.support_dict, sparse=False)
                x = torch.mm(self.support.to(x.device), x)
        else:
            x = torch.mm(self.support.to(x.device), x)

        if self.bias is not None:
            x += self.bias
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'


class PairNorm(Module):
    def __init__(self, mode='PN', scale=1):
        """
            take from https://github.com/LingxiaoShawn/PairNorm/blob/master/layers.py
            (#L85-#L127)
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)

            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x