import torch
from typing import Union, Tuple, Callable, List, Optional
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from layers import NNConv,NNConvAdj, global_sum_pool, GraphConvolution, GraphAggregation, GraphConvolutionWithLabel


class EdgeNet(torch.nn.Module):

    def __init__(self, in_channels: int, layer_out_list: List[int]):
        super(EdgeNet, self).__init__()
        self.in_channels = in_channels
        self.layer_out_list = layer_out_list
        layers = []
        for idx, units in enumerate(self.layer_out_list):
            if idx == 0:
                layers.append(nn.Linear(self.in_channels, self.layer_out_list[idx]))
            else:
                layers.append(nn.Linear(self.layer_out_list[idx - 1], self.layer_out_list[idx]))
            if idx != len(layer_out_list) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, edge_attr):
        x = self.layers(edge_attr)
        return x


class NNConvNet(torch.nn.Module):

    def __init__(self, node_features: int, edge_features: int, conv_layer_out: List[int], dense_layer_out: List[int], edge_nn_out_list: List[int], classification: bool):
        super(NNConvNet, self).__init__()

        self.edge_nn = EdgeNet(edge_features, edge_nn_out_list)
        self.conv_layer_out = conv_layer_out
        self.dense_layer_out = dense_layer_out
        self.node_features = node_features
        self.classification = classification

        layers = []
        for idx, units in enumerate(self.conv_layer_out):
            if idx == 0:
                layers.append(NNConv(self.node_features, self.conv_layer_out[idx], self.edge_nn))
            else:
                layers.append(nn.Linear(self.conv_layer_out[idx - 1], self.conv_layer_out[idx], self.edge_nn))
        self.conv_layers = nn.ModuleList(layers)

        layers = []
        for idx, units in enumerate(self.dense_layer_out):
            if idx == 0:
                layers.append(nn.Linear(self.conv_layer_out[-1], self.dense_layer_out[idx]))
            else:
                layers.append(nn.Linear(self.dense_layer_out[idx-1], self.dense_layer_out[idx]))
        self.dense_layers = nn.ModuleList(layers)

    def forward(self, node_attr, edge_index, edge_attr, batching):

        for layer in self.conv_layers[:-1]:
            node_attr = layer(node_attr, edge_index, edge_attr).clamp(0)
        node_attr = self.conv_layers[-1](node_attr, edge_index, edge_attr)

        x = global_sum_pool(node_attr, batching)

        for layer in self.dense_layers[:-1]:
            x = layer(x).clamp(0)
        x = self.dense_layers[-1](x)

class NNAdjConvNet(torch.nn.Module):

    def __init__(self, node_features: int, edge_features: int, conv_layer_out: List[int], dense_layer_out: List[int], edge_nn_out_list: List[int], non_lin= F.relu):
        super(NNAdjConvNet, self).__init__()


        #self.edge_nn = EdgeNet(edge_features, edge_nn_out_list)
        self.conv_layer_out = conv_layer_out
        self.dense_layer_out = dense_layer_out
        self.node_features = node_features
        self.non_lin = non_lin

        edge_nns = []
        layers = []
        for idx, units in enumerate(self.conv_layer_out):
            if idx == 0:
                edge_nn = EdgeNet(edge_features, edge_nn_out_list + [self.node_features*self.conv_layer_out[idx]])
                layers.append(NNConvAdj(self.node_features, self.conv_layer_out[idx], edge_nn))
            else:
                edge_nn = EdgeNet(edge_features, edge_nn_out_list + [self.conv_layer_out[idx-1] * self.conv_layer_out[idx]])
                layers.append(NNConvAdj(self.conv_layer_out[idx - 1], self.conv_layer_out[idx], edge_nn))
        self.conv_layers = nn.ModuleList(layers)

        layers = []
        for idx, units in enumerate(self.dense_layer_out):
            if idx == 0:
                layers.append(nn.Linear(self.conv_layer_out[-1], self.dense_layer_out[idx]))
            else:
                layers.append(nn.Linear(self.dense_layer_out[idx-1], self.dense_layer_out[idx]))
        self.dense_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(self.dense_layer_out[-1], 1)

    def forward(self, node_attr, edge_adj, activation=None):
        edge_adj =  edge_adj[:,:,:,1:]
        for layer in self.conv_layers[:-1]:
            node_attr = self.non_lin(layer(node_attr, edge_adj))
        node_attr = self.conv_layers[-1](node_attr, edge_adj)

        x = node_attr.sum(dim=-2)

        for layer in self.dense_layers:
            x = self.non_lin(layer(x))
        x = self.output_layer(x)
        x = activation(x) if activation is not None else x
        return x

class GraphGenerator(nn.Module):
    """Generator network from original model"""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(GraphGenerator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1,self.vertexes,self.nodes))

        return edges_logits, nodes_logits




class Triangle_Generator(nn.Module):
    """
    given input vector (typically uniform noise for GAN) outputs upper triangle of edge adjacency
    and node feature matrix.
    :param conv_dims - dimensions of each linear layer (not actually conv)
    :param z_dim - input dimension
    :param vertexes - max number of nodes in graph
    :param edges - edge features
    :param nodes - node features
    :param dropout - dropout percentage
    """

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(Triangle_Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.vec_length = sum(range(vertexes))

        # generate mlp layers
        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=False))
        self.layers = nn.Sequential(*layers)
        # generate outputs
        self.edges_layer = nn.Linear(conv_dims[-1], edges * self.vec_length)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output) \
            .view(-1, self.vec_length, self.edges)
        edges_logits = self.dropoout(edges_logits)

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class OrigDiscriminator(nn.Module):
    """
    basic implementation of original discriminator network
    """
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(OrigDiscriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, m_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h

class FixedDiscriminator(nn.Module):
    """
    basic implementation of original discriminator network
    """
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(FixedDiscriminator, self).__init__()

        # graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        gcn_layers = []
        for idx, u in enumerate(conv_dim):
            graph_conv_dim, aux_dim, linear_dim = u
            if idx == 0:
                last = m_dim
            gcn_layer = GraphConvolution(last, graph_conv_dim, dropout)
            gcn_layers.append(gcn_layer)
            last = graph_conv_dim
        self.gcn_layers = torch.nn.ModuleList(gcn_layers)
        self.agg_layer = GraphAggregation(graph_conv_dim, aux_dim, m_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        for layer in self.gcn_layers:
            annotations = torch.cat((hidden, node), -1) if hidden is not None else node
            h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h

class SimplifiedDiscriminator(nn.Module):
    """
    basic implementation of original discriminator network
    """
    def __init__(self, conv_dim, m_dim, b_dim, dropout, gc_activation=None, dense_activation=None):
        super(SimplifiedDiscriminator, self).__init__()

        # graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        gcn_layers = []
        self.gc_activation = gc_activation
        self.dense_activation = dense_activation
        self.m_dim = m_dim
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        for idx, u in enumerate(graph_conv_dim):
            if idx == 0:
                last = m_dim
            gcn_layer = GraphConvolution(last, u, dropout)
            gcn_layers.append(gcn_layer)
            last = u + m_dim

        self.gcn_layers = torch.nn.ModuleList(gcn_layers)
        self.agg_layer = GraphAggregation(last, aux_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(dense_activation())
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        hidden=None
        for layer in self.gcn_layers:
            annotations = torch.cat((hidden, node), -1) if hidden is not None else node
            hidden = layer(annotations, adj, activation=self.gc_activation)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node

        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h

class SimplifiedDiscriminatorWithLabel(nn.Module):
    """
    basic implementation of original discriminator network
    """
    def __init__(self, conv_dim, m_dim, b_dim, dropout, gc_activation=None, dense_activation=None, global_in=1):
        super(SimplifiedDiscriminatorWithLabel, self).__init__()

        # graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        gcn_layers = []
        self.gc_activation = gc_activation
        self.dense_activation = dense_activation
        self.m_dim = m_dim
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        for idx, u in enumerate(graph_conv_dim):
            if idx == 0:
                last = m_dim
                gcn_layer = GraphConvolutionWithLabel(last, u, dropout,global_in=global_in)
            else:
                gcn_layer = GraphConvolution(last, u, dropout)
            gcn_layers.append(gcn_layer)
            last = u + m_dim

        self.gcn_layers = torch.nn.ModuleList(gcn_layers)
        self.agg_layer = GraphAggregation(last, aux_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(dense_activation())
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, node, label, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        hidden=None

        # special layer
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        hidden = self.gcn_layers[0](annotations, adj, label, activation=self.gc_activation)
        for layer in self.gcn_layers[1:]:
            annotations = torch.cat((hidden, node), -1) if hidden is not None else node
            hidden = layer(annotations, adj, activation=self.gc_activation)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node

        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h