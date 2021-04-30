from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from scipy import sparse

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from utils import load_dict, load_data2tensor, accuracy, row_diff, col_diff
from models import GNN
from pytorchtools import EarlyStopping


def train(model, loss, optimizer, max_epoch, l2, fastmode, patience, features, labels, idx_train, idx_val, chp_name):
    train_result = []
    acc_val_list = []
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chp_name, delta=0.001)
    for epoch in range(max_epoch):
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features)

        loss_train = loss(input=output[idx_train], target=labels[idx_train])
        loss_train = loss_train + l2 * model.loss_spt_param()

        acc_train = accuracy(output=output[idx_train], labels=labels[idx_train])

        loss_train.backward()
        optimizer.step()

        if not fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features)

        loss_val = loss(input=output[idx_val], target=labels[idx_val])
        acc_val = accuracy(output=output[idx_val], labels=labels[idx_val])

        result = dict({
            'Epoch': epoch + 1,
            'loss_train': loss_train.item(),
            'acc_train': acc_train.item(),
            'loss_val': loss_val.item(),
            'acc_val': acc_val.item(),
            'support_learnable': model.get_support_learnable_param,
            # 'row_diff': row_diff(output),
            # 'col_diff': col_diff(output),
            'time': time.time() - t
        })
        train_result.append(result)

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'support: {}'.format(result['support_learnable']),
            #   'row_diff: {:.4f}'.format(result['row_diff']), 
            #   'col_diff: {:.4f}'.format(result['col_diff']),
              'time: {:.4f}s'.format(time.time() - t))
        print('---------------------------------')

        acc_val_list.append(acc_val.item())
        valid_error = 1.0 - acc_val.item()

        scheduler.step()
        early_stopping(valid_error, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return train_result

def test(model, loss, features, labels, idx_test):
    model.eval()
    output = model(features)
    loss_test = loss(input=output[idx_test], target=labels[idx_test])
    acc_test = accuracy(output=output[idx_test], labels=labels[idx_test])

    result = dict({
            'loss_test': loss_test.item(),
            'acc_test': acc_test.item(),
            'support_learnable': model.get_support_learnable_param,
            'row_diff': row_diff(output),
            'col_diff': col_diff(output)})

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          'support: {}'.format(result['support_learnable']),
          'row_diff: {:.4f}'.format(result['row_diff']), 
          'col_diff: {:.4f}'.format(result['col_diff']))

    return result

def inference(state_dict_dir, config_dir, device = 'cuda:0'):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    config = load_dict(config_dir)
    print(config['name'])
    use_cuda = not config['Train']['no_cuda'] and torch.cuda.is_available()

    adj, features, labels, idx_train, idx_val, idx_test = load_data2tensor(config['Train']['dataset'])

    if config['Layers'][-1]['out_dim'] != max(labels).item() + 1:
        config['Layers'][-1]['out_dim'] = max(labels).item() + 1
    config['Layers'][-1]['activation'] = 'none'

    model = GNN(in_dim=features.shape[1], adj = adj, config=config['Layers'])
    state_dict = torch.load(state_dict_dir)
    model.load_state_dict(state_dict)

    if use_cuda:
        device = 'cuda:0' if device is None else device
        model = model.cuda(device)
        model.set_adj_cuda(device)
        features = features.cuda(device)
        labels = labels.cuda(device)
        idx_train = idx_train.cuda(device)
        idx_val = idx_val.cuda(device)
        idx_test = idx_test.cuda(device)

    loss = getattr(torch.nn.functional, 'cross_entropy')

    test_result = test(model=model, loss=loss, features=features, labels=labels, idx_test=idx_test)
    return test_result


def run(config, checkpoint_filename, device = None):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    print(config['name'])
    use_cuda = not config['Train']['no_cuda'] and torch.cuda.is_available()

    np.random.seed(config['Train']['seed'])
    torch.manual_seed(config['Train']['seed'])
    if use_cuda:
        torch.cuda.manual_seed(config['Train']['seed'])

    adj, features, labels, idx_train, idx_val, idx_test = load_data2tensor(config['Train']['dataset'])

    if config['Layers'][-1]['out_dim'] != max(labels).item() + 1:
        config['Layers'][-1]['out_dim'] = max(labels).item() + 1
    config['Layers'][-1]['activation'] = 'none'

    model = GNN(in_dim=features.shape[1], adj = adj, config=config['Layers'])
    # print(model)

    # use_cuda = False
    if use_cuda:
        device = 'cuda:0' if device is None else device
        model = model.cuda(device)
        model.set_adj_cuda(device)
        features = features.cuda(device)
        labels = labels.cuda(device)
        idx_train = idx_train.cuda(device)
        idx_val = idx_val.cuda(device)
        idx_test = idx_test.cuda(device)
    
    optimizer = getattr(torch.optim, config['Optimizer']['mode'])(model.parameters(), lr=config['Optimizer']['lr'], weight_decay=config['Optimizer']['weight_decay'])
    loss = getattr(torch.nn.functional, 'cross_entropy')

    t_total = time.time()

    train_result = train(model = model, 
                        loss = loss,
                        optimizer = optimizer,
                        max_epoch = config['Train']['max_epoch'], 
                        l2 = config['Train']['support_l2'], 
                        fastmode = config['Train']['fastmode'], 
                        patience = config['Train']['patience'], 
                        features = features, 
                        labels = labels,
                        idx_train = idx_train, 
                        idx_val = idx_val,
                        chp_name = checkpoint_filename)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test_result = test(model=model, loss=loss, features=features, labels=labels, idx_test=idx_test)

    return train_result, test_result


def reproducible_run(config, ckp_val, device=None, ckp_init_save=None, ckp_init_load=None, ckp_end=None):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    print(config['name'])
    use_cuda = not config['Train']['no_cuda'] and torch.cuda.is_available()

    np.random.seed(config['Train']['seed'])
    torch.manual_seed(config['Train']['seed'])
    if use_cuda:
        torch.cuda.manual_seed(config['Train']['seed'])

    adj, features, labels, idx_train, idx_val, idx_test = load_data2tensor(config['Train']['dataset'])

    if config['Layers'][-1]['out_dim'] != max(labels).item() + 1:
        config['Layers'][-1]['out_dim'] = max(labels).item() + 1
    config['Layers'][-1]['activation'] = 'none'

    model = GNN(in_dim=features.shape[1], adj = adj, config=config['Layers'])

    if ckp_init_load is not None:
        state_dict = torch.load(ckp_init_load)
        state_dict['GraphConvolution0.alpha'] = torch.FloatTensor(config['Layers'][0]['support']['alpha'])
        state_dict['GraphConvolution0.beta'] = torch.FloatTensor(config['Layers'][0]['support']['beta'])
        state_dict['GraphConvolution1.alpha'] = torch.FloatTensor(config['Layers'][1]['support']['alpha'])
        state_dict['GraphConvolution1.beta'] = torch.FloatTensor(config['Layers'][1]['support']['beta'])
        model.load_state_dict(state_dict)

    if ckp_init_save is not None:
        torch.save(model.state_dict(), ckp_init_save)

    # use_cuda = False
    if use_cuda:
        device = 'cuda:0' if device is None else device
        model = model.cuda(device)
        model.set_adj_cuda(device)
        features = features.cuda(device)
        labels = labels.cuda(device)
        idx_train = idx_train.cuda(device)
        idx_val = idx_val.cuda(device)
        idx_test = idx_test.cuda(device)

    optimizer = getattr(torch.optim, config['Optimizer']['mode'])(model.parameters(), lr=config['Optimizer']['lr'], weight_decay=config['Optimizer']['weight_decay'])
    loss = getattr(torch.nn.functional, 'cross_entropy')

    t_total = time.time()

    train_result = train(model = model, 
                        loss = loss,
                        optimizer = optimizer,
                        max_epoch = config['Train']['max_epoch'], 
                        l2 = config['Train']['support_l2'], 
                        fastmode = config['Train']['fastmode'], 
                        patience = config['Train']['patience'], 
                        features = features, 
                        labels = labels,
                        idx_train = idx_train, 
                        idx_val = idx_val,
                        chp_name = ckp_val)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    if ckp_end is not None:
        torch.save(model.state_dict(), ckp_end)

    # Testing
    test_result = test(model=model, loss=loss, features=features, labels=labels, idx_test=idx_test)
    return train_result, test_result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/QPGCN-cora.json", help='Config(json) to use.')
    parser.add_argument('--result', type=str, default="./result/", help='The directory of the result.')
    parser.add_argument('--device', type=str, default="cuda:0", help='The device to use.')
    args = parser.parse_args()

    config = load_dict(args.config)
    train_result, test_result = run(config, args.result + 'checkpoint.pt', args.device)
