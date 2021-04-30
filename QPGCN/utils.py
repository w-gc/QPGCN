import json
import torch
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
import pickle as pkl
import networkx as nx
import sys
import os


def load_dict(filename):
    r"""load dict from json file"""
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic



def parse_index_file(filename):
    r"""
        Parse index file.
        take from https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def load_data(dataset_str):
    r"""
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        take from https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py

        Args:
            dataset_str:        Dataset name
        Returns:
            adj:                scipy.sparse.csr.csr_matrix, the Adjacency matrix
            features:           scipy.sparse.lil.lil_matrix, the feature matrix of the nodes
            labels:             numpy.ndarray, labels[1]: [0, 0, 0, 0, 1, 0, 0], one-shot
            idx_train:          range
            idx_val:            range
            idx_test:           range
    """
    dataset_str = dataset_str.lower()
    if dataset_str == 'dblp':
        return load_dblp_data()

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test



def load_data2tensor(dataset_str):
    r"""
        the output of the load_data() is converted to tensor
        Args:
            dataset_str:        Dataset name
        Returns:
            adj:                scipy.sparse.csr.csr_matrix
            features:           torch.Tensor
            labels:             torch.LongTensor, Remove the one-shot encoding
            idx_train:          tensor([...])
            idx_val:            tensor([...])
            idx_test:           tensor([...])
    """

    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_str)
    adj = sparse_mx_to_torch_sparse_tensor(adj)                 # <class 'scipy.sparse.csr.csr_matrix'> --> torch sparse tensor
    features = normalize(features)                              # <class 'scipy.sparse.lil.lil_matrix'> --> <class 'scipy.sparse.csr.csr_matrix'>
    features = torch.FloatTensor(np.array(features.todense()))  # <class 'scipy.sparse.csr.csr_matrix'> --> <class 'torch.Tensor'>
    labels = torch.LongTensor(np.argmax(labels, axis = 1))      # <class 'numpy.ndarray'> one-shot --> label_num
    
    idx_train = torch.LongTensor(idx_train)                     # <class 'numpy.ndarray'>
    idx_val = torch.LongTensor(idx_val)                         # <class 'numpy.ndarray'>
    idx_test = torch.LongTensor(idx_test)                       # <class 'numpy.ndarray'>

    return adj, features, labels, idx_train, idx_val, idx_test



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    r"""
        Convert a scipy sparse matrix to a torch sparse tensor.
        take from https://github.com/dms-net/scatteringGCN/blob/master/utils.py
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    r"""
        Row-normalize sparse matrix.
        take from https://github.com/dms-net/scatteringGCN/blob/master/utils.py
    """
    rowsum = np.array(mx.sum(1)) 
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    r"""
        Calculate the accuracy
        Args:
            output:             the output of the GNN, one-shot encoding
            labels:             ground truth label
        Returns:
            acc.:                 
    """
    preds = output.argmax(1).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_dblp_data():
    r"""
        the dataset of DBLP take from https://github.com/dms-net/scatteringGCN/blob/master/DBLP/DBLP_GCN.py
    """

    dataset_str = './data/dblp.npy'

    if not os.path.exists(dataset_str):
        print('-----------dblp downloading--------------------')
        from torch_geometric.datasets import CitationFull
        import torch_geometric.transforms as T
        # https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz
        dataset = CitationFull('./data/dblp', name='dblp',transform=T.TargetIndegree())
        np.save(dataset_str, {'edge_attr' : dataset[0].edge_attr.numpy(),
                                'edge_index' : dataset[0].edge_index.numpy(),
                                'x' : dataset[0].x.numpy(),
                                'y' : dataset[0].y.numpy(),
                                'idx_train': range(0, 10000),
                                'idx_val': range(10000, 12000),
                                'idx_test': range(12000, 14000)})
    
    data = np.load(dataset_str, allow_pickle=True).item() # readout

    nodes_num = data['x'].shape[0]
    edges_num = data['edge_attr'].size
    idx_shuffle = np.array(range(nodes_num))
    np.random.shuffle(idx_shuffle)
    adj = sp.csr_matrix((np.ones(edges_num), (data['edge_index'][0,:], data['edge_index'][1,:])), shape=(nodes_num, nodes_num))
    adj = adj[idx_shuffle, :][:, idx_shuffle]

    labels = np.zeros( (nodes_num, max(data['y']) - min(data['y']) + 1) )
    for i, label in enumerate(data['y'][idx_shuffle]):
        labels[i, label] = 1
    features = sp.csr_matrix(data['x'][idx_shuffle,:]).tolil()
    idx_train = data['idx_train']
    idx_val = data['idx_val']
    idx_test = data['idx_test']

    return adj, features, labels, idx_train, idx_val, idx_test

@torch.no_grad()
def row_diff(x):
    n = x.shape[0]
    rd = 0.0
    for i in range(n):
        rd += torch.norm((x[i] - x), p=2, dim=1).mean()
    return (rd / n).item()

@torch.no_grad()
def col_diff(x):
    return row_diff( (x / torch.norm(x, p=1, dim=0)).T )

@torch.no_grad()
def DirichletEnergy(x, add):
    from support_utils import normalize_adj
    N = adj.shape[0]
    L = torch.eye(N) - normalize_adj(adj, gamma=0.0, sparse=False)
    return 2 * x.T.mm(torch.mm(L.to(x.device), x))