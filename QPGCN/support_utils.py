import json
import torch
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
import pickle as pkl
import networkx as nx
import sys

def pre_support(adj, sup_dict, sparse=True):

    if sup_dict['name'] == 'none':
        support = Identity(adj.shape[0], sparse=sparse, Dev=adj.device)

    elif sup_dict['name'] == 'Adj':
        support = Adj_power(adj, sup_dict['power'], sparse=sparse)

    elif sup_dict['name'] == 'NormAdj':
        support = Adj_power(normalize_adj(adj), sup_dict['power'], sparse=sparse)

    elif sup_dict['name'] == 'gNormAdj':
        support = Adj_power(normalize_adj(adj, sup_dict['gamma']), sup_dict['power'], sparse=sparse)
    
    elif sup_dict['name'] == 'Adj_tilde':
        support = Adj_tilde(adj, sup_dict['gamma'])

    elif sup_dict['name'] == 'RandomWalk':
        support = Adj_power(RandomWalk(adj), sup_dict['power'], sparse=sparse)
    
    elif sup_dict['name'] == 'LazyRandomWalk':
        support = Adj_power(LazyRandomWalk(adj), sup_dict['power'], sparse=sparse)

    elif sup_dict['name'] == 'Scatter':
        support = Adj_power(scattering(adj, sup_dict['path']), sup_dict['power'], sparse=sparse)

    elif sup_dict['name'] == 'ResConv':
        support = ResConv(adj, sup_dict['alpha'], sparse=sparse)

    elif sup_dict['name'] == 'Chebyshev':
        support = chebyshev_polynomials(adj, sup_dict['k'], sparse=sparse)
    
    elif sup_dict['name'] == 'QPF':
        if sparse:
            support = Adj_power(QPF(adj, gamma=sup_dict['gamma'], alpha=sup_dict['alpha'], beta=sup_dict['beta']), power = sup_dict['power'], sparse=sparse)
        else:
            support = QPF(adj, gamma=sup_dict['gamma'], alpha=sup_dict['alpha'], beta=sup_dict['beta'], sparse=False).matrix_power(sup_dict['power'])
    
    elif sup_dict['name'] == 'DeltaQPF':
        support = DeltaQPF(A_tilde=adj, degrees=sup_dict['degrees'], C=sup_dict['C'], gamma=sup_dict['gamma'], alpha=sup_dict['alpha'], beta=sup_dict['beta'], sparse=sparse)

    elif sup_dict['name'] == 'Exponential':
        support = exponential(adj, gamma=sup_dict['gamma'], mu=sup_dict['mu'], sigma=sup_dict['sigma']).matrix_power(sup_dict['power'])
    
    else:
        raise ValueError("The support method '{}' has NOT been implemented yet ".format(sup_dict['name']))

    return support

def Identity(N, C = 1, sparse=True, Dev ='cpu'):
    if not sparse:
        return C * torch.eye(N)
    return torch.sparse.FloatTensor( torch.LongTensor((range(N), range(N))).to(Dev), C * torch.ones(N).to(Dev),  torch.Size([N, N]) )

def Adj_power(adj, power = 1, sparse=True):
    if power == 1:
        return adj
    if adj.is_sparse:
        adj = adj.to_dense()
    adj_p = torch.matrix_power(adj, power)
    if not sparse:
        return adj_p
    return adj_p.to_sparse()

def Adj_tilde(adj, gamma = 1.0, sparse=True):
    r"""
        Symmetrically normalize adjacency matrix.
        :math: `P_{\gamma} = ( D + \gamma I_N )^{-1/2}'
        :math: `\tilde{A} = P_{\gamma} A P_{\gamma}`
    """
    N = adj.shape[0]
    degrees = torch.sparse.sum(adj, dim = 1).to_dense()
    degrees += gamma * torch.ones(adj.shape[0]).to()
    d_inv_sqrt = degrees ** (-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    A_tilde = spmMdiagm(spmMdiagm(adj, d_inv_sqrt).transpose(1, 0), d_inv_sqrt).transpose(1, 0) 
    if not sparse:
        return A_tilde.to_dense()
    return A_tilde

def normalize_adj(adj, gamma = 1.0, sparse=True):
    r"""
        Symmetrically normalize adjacency matrix.
        :math: `D^{-1/2} A D^{-1/2}`
    """
    N = adj.shape[0]
    degrees = torch.sparse.sum(adj, dim = 1).to_dense()
    degrees += gamma * torch.ones(adj.shape[0]).to()
    d_inv_sqrt = degrees ** (-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    norm_adj = spmMdiagm(spmMdiagm(adj, d_inv_sqrt).transpose(1, 0), d_inv_sqrt).transpose(1, 0) \
                + torch.sparse.FloatTensor( torch.LongTensor((range(N), range(N))).to(adj.device), gamma * d_inv_sqrt ** 2,  torch.Size([N, N]) )

    if not sparse:
        return norm_adj.to_dense()
    return norm_adj

def RandomWalk(adj, sparse=True):
    r"""
        Random walk matrix
        :math: `A D^{-1}`
        take from https://github.com/dms-net/scatteringGCN/blob/master/utils.py
    """
    degrees = torch.sparse.sum(adj, dim = 1).to_dense()
    d_inv = degrees ** (-1)
    d_inv[torch.isinf(d_inv)] = 0.
    rw = spmMdiagm(adj, d_inv)
    if not sparse:
        return rw.to_dense()
    return rw

def LazyRandomWalk(adj, sparse=True):
    r"""
        Lazy Random walk matrix
        :math: `0.5 * (I_N + A D^{-1})`
        take from https://github.com/dms-net/scatteringGCN/blob/master/utils.py
    """
    rw = RandomWalk(adj, sparse=sparse)
    if not sparse:
        return 0.5 * ( torch.eye(adj.shape[0]).to(adj.device) + rw )
    values = 0.5 * ( rw.coalesce().values() + torch.ones(adj.shape[0]) )
    return torch.sparse.FloatTensor(adj.coalesce().indices(), values,  adj.shape)

def scattering1st(lrw, order, sparse = True):
    r"""
        Scatter transformation matrix
        :math: `P = 0.5*(I_N + AD^{-1})`
        :math: `\Psi_{0} = I_N - P`
        :math: `\Psi_{k} = p^{2^{k-1}} (I_N - p^{2^{k-1}}), k \geq 1`
        take from https://github.com/dms-net/scatteringGCN/blob/master/utils.py
        Args:
            lrw:            :math: `P = 0.5*(I_N + AD^{-1})`， lazy Random walk matrix
            order:          :math: `k`
        Returns:
            adj_int:        :math: `\Psi_{k}`
    """
    assert order >= 0, 'the order of the scattering matrix MUST be nonnegative'

    I = torch.eye(lrw.shape[0])
    
    if order == 0:
        adj_int = I - lrw
    else:
        # `\Psi_{k} = p^{2^{k-1}} (I_N - p^{2^{k-1}}), k \geq 1`
        adj_power = lrw.matrix_power(2**(order-1))
        adj_int = (adj_power - I).mm(adj_power)
    
    if not sparse:
        return adj_int
    return adj_int.to_sparse()

def scattering(adj, path, sparse = True):
    sct = torch.eye(adj.shape[0])
    lrw = LazyRandomWalk(adj, sparse=False)
    for i in path:
        sct = sct.mm(scattering1st(lrw, i, sparse = False))
    if not sparse:
        return sct
    return sct.to_sparse()

def ResConv(adj, alpha=1.0, sparse=True):
    r"""
        :math: $\frac{1}{1 + \alpha} (I_N + \alpha A D^{-1})$
    """
    rw = RandomWalk(adj, sparse=sparse)
    if not sparse:
        return 1/(1 + alpha) * ( alpha * rw + torch.eye(adj.shape[0]) )
    return 1/(1 + alpha) * ( alpha * rw + Identity(adj.shape[0]) )

def chebyshev_polynomials(adj, k, sparse=True):
    r"""
        Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    from scipy.sparse.linalg.eigen.arpack import eigsh
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    adj_normalized = normalize_adj(adj, gamma=0.0, sparse=False)

    laplacian = torch.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian.numpy(), 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - torch.eye(adj.shape[0])

    if k == 0:
        return Identity(adj.shape[0], sparse, Dev=adj.device)
    elif k == 1:
        if not sparse:
            return scaled_laplacian
        return scaled_laplacian.to_sparse()

    t_k_2 = torch.eye(adj.shape[0])
    t_k_1 = scaled_laplacian
    for i in range(2, k+1):
        t_k = 2 * scaled_laplacian.mm(t_k_1) - t_k_2
        t_k_2 = t_k_1
        t_k_1 = t_k
    
    if not sparse:
        return t_k
    return t_k.to_sparse()

def spmMdiagm(spm, diagm):
    values = diagm[spm.coalesce().indices()[1,:]] * spm.coalesce().values()
    return torch.sparse.FloatTensor(spm.coalesce().indices(), values,  spm.shape)


def DeltaQPF(A_tilde, degrees, C, gamma=1.0, alpha=0., beta=2., sparse = True):
    beta = alpha + beta
    N = A_tilde.shape[0]
    Dev = A_tilde.device
 
    D_gamma = degrees + gamma * torch.ones(N).to(Dev)
    P_gamma = D_gamma ** (-0.5)
    P_gamma[torch.isinf(P_gamma)] = 0.

    P_alpha_tilde = P_gamma * ( (1 - alpha) * gamma * torch.ones(N).to(Dev) - alpha * degrees ) * P_gamma
    P_beta_tilde = P_gamma * ( (1 - beta) * gamma * torch.ones(N).to(Dev) - beta * degrees ) * P_gamma
    
    indices = torch.LongTensor((range(N), range(N))).to(Dev)
    part_sum = torch.sparse.FloatTensor(indices, C * P_alpha_tilde * P_beta_tilde,  A_tilde.shape)
    part_sum += spmMdiagm(A_tilde.transpose(1, 0), C * P_alpha_tilde).transpose(1, 0)
    part_sum += spmMdiagm(A_tilde, C * P_beta_tilde)
    torch.cuda.empty_cache()
    if not sparse:
        return part_sum.to_dense()
    return part_sum

def QPF(adj, gamma=1., alpha=0., beta=2., sparse = True):
    r"""
        :math: `S_{\gamma} - \alpha I_N = ( D + \gamma I_N )^{-1/2} ( A + \gamma I_N) ( D + \gamma I_N )^{-1/2} - \alpha I_N := P_{\gamma} (A + P_{\alpha}) P_{\gamma}`
        :math: `(S_{\gamma} - \alpha I_N )(S_{\gamma} - \beta I_N ) = \tilde{A} \tilde{A} + \tilde{A} \tilde{P}_{\beta} + \tilde{P}_{\alpha} \tilde{A} + \tilde{P}_{\alpha} \tilde{P}_{\beta}`
    """
    beta = alpha + beta
    N = adj.shape[0]
    Dev = adj.device
    C = -1

    D = torch.sparse.sum(adj, dim = 1).to_dense()

    # $D_{\gamma} = D + \gamma I_N$
    D_gamma = D + gamma * torch.ones(N).to(Dev)

    # P_{\gamma} = ( D + \gamma I_N )^{-1/2}
    P_gamma = D_gamma ** (-0.5)
    P_gamma[torch.isinf(P_gamma)] = 0.

    # $\tilde{A} = P_{\gamma} A P_{\gamma}$
    A_tilde = spmMdiagm(spmMdiagm(adj, P_gamma).transpose(1, 0), P_gamma).transpose(1, 0)

    # $P_{\alpha} = (1 - \alpha) \gamma I_N - \alpha D$
    # $\tilde{P}_{\alpha} = P_{\gamma} P_{\alpha} P_{\gamma}$
    P_alpha_tilde = P_gamma * ( (1 - alpha) * gamma * torch.ones(N).to(Dev) - alpha * D ) * P_gamma

    # $P_{\beta} = (1 - \beta) \gamma I_N - \beta D$
    # $\tilde{P}_{\beta} = P_{\gamma} P_{\beta}A P_{\gamma}$
    P_beta_tilde = P_gamma * ( (1 - beta) * gamma * torch.ones(N).to(Dev) - beta * D ) * P_gamma
    

    # $+ \tilde{P}_{\alpha} \tilde{P}_{\beta}$
    indices = torch.LongTensor((range(N), range(N))).to(Dev)
    part_sum = torch.sparse.FloatTensor(indices, C * P_alpha_tilde * P_beta_tilde,  adj.shape)

    # $+ \tilde{P}_{\alpha} \tilde{A}$
    part_sum += spmMdiagm(A_tilde.transpose(1, 0), C * P_alpha_tilde).transpose(1, 0)

    # $+ \tilde{A} \tilde{P}_{\beta}$
    part_sum += spmMdiagm(A_tilde, C * P_beta_tilde)

    # $+ \tilde{A} \tilde{A}$
    if not sparse:
        return part_sum.to_dense() + torch.sparse.mm(A_tilde, C * A_tilde.to_dense())
    return part_sum + torch.sparse.mm(A_tilde, C * A_tilde.to_dense()).to_sparse()