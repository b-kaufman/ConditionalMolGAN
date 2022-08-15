import torch
from scipy.linalg import eigvals
from utils import postprocess
import numpy as np
import random

def structural_eigen_similarity(adj1,adj2,processed=False,dist='l1'):
    #must be symmetric (no gumbel)
    if not processed:
        adj1, adj2 = postprocess((adj1, adj2), 'softmax')
    eigval1 = torch.symeig(adj1.permute(2,0,1).contiguous(), eigenvectors=True).eigenvalues
    eigval2 = torch.symeig(adj2.permute(2,0,1).contiguous(), eigenvectors=True).eigenvalues
    if dist == 'l1':
        return torch.sum(torch.abs(eigval1-eigval2))
    if dist == 'l2':
        return torch.sum(torch.norm(eigval1-eigval2, dim=1))

def KL_pairwise_similarity(adj1,adj2,node1,node2):
    #DKL(adj1||adj2)
    KL_adj = torch.nn.functional.kl_div(torch.log(adj1), torch.log(adj2), log_target=True, reduction='batchmean')
    KL_node = torch.nn.functional.kl_div(torch.log(node1), torch.log(node2), log_target=True, reduction='batchmean')
    return KL_adj + KL_node

def JS_pairwise_similarity(adj1,adj2,node1,node2):
    #DKL(adj1||adj2)
    JS_adj = .5 * (torch.nn.functional.kl_div(torch.log(adj1), torch.log(adj2), log_target=True, reduction='batchmean') + \
                   torch.nn.functional.kl_div(torch.log(adj2), torch.log(adj1), log_target=True, reduction='batchmean'))
    JS_node = .5 * (torch.nn.functional.kl_div(torch.log(node1), torch.log(node2), log_target=True, reduction='batchmean') + \
              torch.nn.functional.kl_div(torch.log(node2), torch.log(node1), log_target=True, reduction='batchmean'))

    return JS_adj + JS_node

def molecular_tanimoto_similarity_from_graph(adj1,node1,adj2,node2):
    pass

def molecular_tanimoto_similarity_from_mols(mol1,mol2):
    pass

def pairwise_batch_comparison(adj_batch,node_batch, similarity_func_name, processed=False, reduction='sum',sample=None):
    if not processed:
        adj_batch, node_batch = postprocess((adj_batch, node_batch), 'softmax')

    pairs = get_pairs(adj_batch.size(0))
    if sample is not None:
        pairs = random.sample(pairs, sample)
    if similarity_func_name == "eig_sim_l1":
        pair_similarities = torch.stack([structural_eigen_similarity(adj_batch[pair[0]], adj_batch[pair[1]],dist='l1') for pair in pairs])
    if similarity_func_name == "eig_sim_l2":
        pair_similarities = torch.stack([structural_eigen_similarity(adj_batch[pair[0]], adj_batch[pair[1]], dist='l2') for pair in
                             pairs])
    if similarity_func_name == "kl_div":
        pair_similarities = torch.stack([KL_pairwise_similarity(adj_batch[pair[0]], adj_batch[pair[1]],
                                                    node_batch[pair[0]], node_batch[pair[1]] ) for pair in pairs])
    if similarity_func_name == "js_div":
        pair_similarities = torch.stack([JS_pairwise_similarity(adj_batch[pair[0]], adj_batch[pair[1]],
                                                    node_batch[pair[0]], node_batch[pair[1]] ) for pair in pairs])
    else:
        ValueError('not acceptable function name')

    if reduction == 'sum':
        return torch.sum(pair_similarities)
    if reduction == 'mean':
        return torch.mean(pair_similarities)
    else:
        ValueError('not acceptable reduction method')


def get_pairs(batch_size):
    pairs = []
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            pairs.append((i,j))
    return pairs

if __name__ == "__main__":
    import networks
    G = networks.GraphGenerator([64, 64, 64], 8, 9, 4, 8, 0)

    z = torch.rand([256, 8])

    e, n = G(z)

    pairwise_batch_comparison(e, n, 'kl_div', sample=64)
