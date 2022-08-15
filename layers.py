from utils import broadcast, scatter_sum, map_nodes_to_edge_index

import torch
from typing import Union, Tuple, Callable, List, Optional
from torch.nn import Parameter
import torch.nn as nn
import math


class NNConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, net: Callable, root_weight: bool = True,
                 bias: bool = True):
        super(NNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.net = net

        assert self.net.layers[-1].out_features == self.in_channels * self.out_channels

        if root_weight:
            self.root = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        if self.root is not None:
            bound = 1.0 / math.sqrt(self.root.size(0))
            if self.root is not None:
                self.root.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, node_attr, edge_index, edge_attr):
        size = node_attr.size()
        scat_node_attr = map_nodes_to_edge_index(node_attr, edge_index, 1)
        weights = self.net(edge_attr)
        weights = weights.view(-1, self.in_channels, self.out_channels)
        messages = torch.matmul(scat_node_attr.unsqueeze(1), weights).squeeze(1)
        output = scatter_sum(messages, edge_index[1], dim=-2, out=torch.zeros(node_attr.size(-2), self.out_channels).to())
        if self.root is not None:
            output = output + torch.matmul(node_attr, self.root)
        return output


class NNConvAdj(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, net: Callable, root_weight: bool = True,
                 bias: bool = True):
        super(NNConvAdj, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.net = net

        assert self.net.layers[-1].out_features == self.in_channels * self.out_channels

        if root_weight:
            self.root = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        if self.root is not None:
            bound = 1.0 / math.sqrt(self.root.size(0))
            if self.root is not None:
                self.root.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, node_attr, edge_adj, activation=None):

        # make sure number of nodes are consistent
        # node_attr batch x num_nodes x features. edge_adj batch x num_nodes x num_nodes x features
        assert node_attr.size(-2) == edge_adj.size(-2)
        if node_attr.dim() == 2:
            node_attr = node_attr.unsqueeze(0)
        if edge_adj.dim() == 3:
            edge_adj.unsqueeze(0)
        batch_size = edge_adj.size(0)
        full_connection_edge_index = torch.arange(0, node_attr.size(-2), device=node_attr.device).repeat(node_attr.size(-2)).unsqueeze(0)

        edge_adj = edge_adj.reshape(edge_adj.size(0), edge_adj.size(-2)*edge_adj.size(-2), -1)
        weights = self.net(edge_adj)
        weights = weights.view(edge_adj.size(0), -1, self.in_channels, self.out_channels)

        scat_node_attr = map_nodes_to_edge_index(node_attr, full_connection_edge_index, 0)
        # messages = torch.matmul(scat_node_attr.unsqueeze(1), weights).squeeze(1)
        messages = torch.einsum('ijk,ijkl->ijl', scat_node_attr, weights)
        output = scatter_sum(messages, full_connection_edge_index[0], dim=-2,
                             out=torch.zeros([batch_size, node_attr.size(-2), self.out_channels], device=node_attr.device))

        if self.root is not None:
            output = output + torch.matmul(node_attr, self.root)

        output = activation(output) if activation is not None else output

        return output

class GraphConvolution(torch.nn.Module):
    """
    basic graph conv layer, each edge type is assigned its own set of weights
    which are applied during message passing.
    """
    def __init__(self, in_features, out_features,  dropout, edge_feat=4):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weights for each edge type
        self.edge_weights = torch.nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(edge_feat)])

        # weights for root
        self.lin_add = nn.Linear(in_features, out_features)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, adj, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9

        # apply each set of edge weights to node_attr
        hidden = torch.stack([filter(node) for filter in self.edge_weights], 1)

        # mask
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))

        # sum messages and root
        hidden = torch.sum(hidden, 1) + self.lin_add(node)

        # non-linearity
        hidden = activation(hidden) if activation is not None else hidden

        #dropout
        hidden = self.dropout(hidden)

        return hidden

class GraphConvolutionWithLabel(torch.nn.Module):
    """
    basic graph conv layer, each edge type is assigned its own set of weights
    which are applied during message passing.
    """
    def __init__(self, in_features, out_features,  dropout, edge_feat=4, global_in=1):
        super(GraphConvolutionWithLabel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weights for each edge type
        self.edge_weights = torch.nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(edge_feat)])
        self.label_weights = nn.Linear(global_in, out_features)
        # weights for root
        self.lin_add = nn.Linear(in_features, out_features)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, adj, label, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9

        # apply each set of edge weights to node_attr
        hidden = torch.stack([filter(node) for filter in self.edge_weights], 1)

        # mask
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))

        # sum messages and root

        hidden = torch.sum(hidden, 1) + self.lin_add(node) + activation(self.label_weights(label).unsqueeze(1))
        # non-linearity
        hidden = activation(hidden) if activation is not None else hidden

        #dropout
        hidden = self.dropout(hidden)

        return hidden


class GraphAggregation(torch.nn.Module):

    """
    original graph aggregation

    """
    def __init__(self, in_features, out_features, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output


def global_sum_pool(x, batch_mat):
    return torch.mm(batch_mat, x)