import torch
from torch.utils.data import Dataset, DataLoader
from networks import OrigDiscriminator
from sparse_molecular_dataset import SparseMolecularDataset
import numpy as np
import utils


class MolPadDataset(Dataset):
    """
    produces padded representation of input graphs
    """
    def __init__(self, root, transform=None, pad_to=50):

        info = torch.load(root)
        self.dataset = info['t_list']
        self.a_decode = info['a_decode']
        self.b_decode = info['b_decode']
        self.transform = transform
        self.pad_to = pad_to

        self._pad_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is list:
            idx = idx[0]

        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        reduced_data = {'node_attr': data['node_attr'],
                        'edge_adj_mat': data['edge_feat_adj_mat'].permute(1, 2, 0),
                        'y': data['y']}

        return reduced_data

    def _pad_data(self):

        for idx, data in enumerate(self.dataset):
            node_padder = torch.nn.ZeroPad2d((0,0,0,self.pad_to-data['node_attr'].size(-2)))
            padded_node_attr = node_padder(data['node_attr'])
            edge_pad = self.pad_to - data['edge_feat_adj_mat'].size(-2)
            edge_padder = torch.nn.ZeroPad2d((0,edge_pad,0,edge_pad))
            padded_edge_attr = edge_padder(data['edge_feat_adj_mat'])
            self.dataset[idx]['node_attr'] = padded_node_attr
            self.dataset[idx]['edge_feat_adj_mat'] = padded_edge_attr

def one_hot_empty_append(data):
    """

    transformation to address the absence of an atom in the fixed size graph
    use when empty channel not already added
    """
    reduced = torch.sum(data['edge_feat_adj_mat'], dim = 0)
    reduced[reduced == 1.] = 2.
    reduced[reduced == 0.] = 1.
    reduced[reduced == 2.] = 0.
    new_edge= torch.cat([reduced.unsqueeze(0), data['edge_feat_adj_mat']])
    reduced = torch.sum(data['node_attr'], dim=1)
    reduced[reduced == 1.] = 2.
    reduced[reduced == 0.] = 1.
    reduced[reduced == 2.] = 0.
    new_node = torch.cat([reduced.unsqueeze(1), data['node_attr']],1)
    return {'node_attr': new_node, 'edge_feat_adj_mat': new_edge, 'y': data['y']}

def one_hot_empty_augment(data):
    '''when empty channel exists'''
    reduced = torch.sum(data['edge_feat_adj_mat'], dim = 0)
    reduced[reduced == 1.] = 2.
    reduced[reduced == 0.] = 1.
    reduced[reduced == 2.] = 0.
    new_edge = data['edge_feat_adj_mat']
    new_edge[0,:,:] = new_edge[0, :, :] + reduced
    reduced = torch.sum(data['node_attr'], dim=1)
    reduced[reduced == 1.] = 2.
    reduced[reduced == 0.] = 1.
    reduced[reduced == 2.] = 0.
    new_node = data['node_attr']
    new_node[:, 0] = new_node[:, 0] + reduced
    return {'node_attr': new_node, 'edge_feat_adj_mat': new_edge, 'y': data['y']}

class SparseMolDataset(Dataset):
    """
    produces padded representation of input graphs
    """
    def __init__(self, root, transform=None, label_map=None):

        self.dataset = SparseMolecularDataset()
        self.dataset.load(root)
        self.transform = transform
        self.adj_size = self.dataset.data_A.shape[-1]
        self.node_depth = self.dataset.atom_num_types
        self.edge_depth = self.dataset.bond_num_types

        if label_map is None:
            self.labels = None
        else:
            self.labels = [k for k in label_map]
            self.dataset._transform_labels(label_map)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is list:
            idx = idx[0]

        edge_adj_mat = self.dataset.one_hot(self.dataset.data_A[idx], depth=self.edge_depth)
        node_attr = self.dataset.one_hot(self.dataset.data_X[idx], depth=self.node_depth)
        # cludge not needed right now
        if self.labels is None or self.labels == []:
            y = np.zeros(1,)
        else:
            y = np.concatenate([self.dataset.label_data[k][idx] for k in self.labels])
        if self.transform:
            data = self.transform(data)

        reduced_data = {'node_attr': torch.Tensor(node_attr),
                        'edge_adj_mat': torch.Tensor(edge_adj_mat),
                        'y': torch.Tensor(y)}

        return reduced_data


if __name__ == '__main__':
    '''
    a = MolPadDataset('data/gdb9.pt',transform=one_hot_empty_augment,pad_to=10)
    test = DataLoader(a, batch_size=10)
    D = OrigDiscriminator([[128, 64], 128, [128, 64]], len(a.a_decode), len(a.b_decode), 0)
    for b in test:
        D(b['edge_adj_mat'], None, b['node_attr'])
        break
    '''
    data = SparseMolecularDataset()
    data.load('data/gdb9_9nodes.sparsedataset')
    dset = SparseMolDataset('data/gdb9_9nodes.sparsedataset',
                            label_map={'logp': 'norm', 'mw': 'bin_5'})

    miss = 0
    for ii in range(data.data_A.shape[0]):
        t = dset[ii]['y']
        a = t
        '''
        t = t.detach().numpy()
        t_shap = t.shape
        res = data.one_hot(data.data_X[ii,:], 5)
        shap = res.shape
        if not np.array_equal(t,res):
            print(ii)
        '''