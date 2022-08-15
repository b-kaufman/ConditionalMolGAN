from networks import GraphGenerator, SimplifiedDiscriminator, NNAdjConvNet, SimplifiedDiscriminatorWithLabel
from similarity import pairwise_batch_comparison
import torch
import torch.nn.functional as F
import numpy as np
import time, datetime
from utils import basic_scores, conditional_basic_scores, get_embedding, get_activation_function, get_init_function, \
    postprocess
from rdkit import Chem
import os
import argparse
from abc import ABC, abstractmethod


class GAN_Trainer(ABC):

    def __init__(self, model_params, device):

        self.params = model_params
        self.step_info = {}
        self.eval_history = {}
        self.epochs_trained = 0
        self.device = device
        self.bond_decoder = model_params.bond_decoder
        self.atom_decoder = model_params.atom_decoder
        self.G, self.D = self.build_models()
        self.init_models()
        self.g_optimizer, self.d_optimizer = self.get_optimizers()
        self.G.to(self.device)
        self.D.to(self.device)
        self.times = []

    def init_models(self):
        init = get_init_function(self.params.layer_init)
        if init is not None:
            self.G.apply(init)
            self.D.apply(init)

    def build_models(self):
        g_name = self.params.generator_type
        d_name = self.params.discriminator_type

        if g_name == 'GraphGenerator':
            full_z = self.params.z_dim

            G = GraphGenerator(self.params.g_conv_dim, full_z, self.params.vertexes, self.params.bond_num_types,
                               self.params.atom_num_types, self.params.dropout)
        elif g_name == 'GraphGeneratorConditional':
            full_z = self.params.z_dim + self.params.label_length

            G = GraphGenerator(self.params.g_conv_dim, full_z, self.params.vertexes, self.params.bond_num_types,
                               self.params.atom_num_types, self.params.dropout)
        else:
            raise ValueError('invalid generator name')

        if d_name == 'SimplifiedDiscriminator':
            dim_group = [self.params.d_conv_dim, self.params.aux_dim, self.params.lin_dim]
            len_a = len(self.atom_decoder)
            D = SimplifiedDiscriminator(dim_group, len_a, len(self.bond_decoder),
                                        self.params.dropout,
                                        gc_activation=get_activation_function(self.params.d_gc_activation),
                                        dense_activation=get_activation_function(self.params.d_dense_activation))
        elif d_name == 'NNAdjConvNet':
            D = NNAdjConvNet(self.params.atom_num_types, self.params.bond_num_types - 1, self.params.d_conv_dim,
                             self.params.lin_dim,
                             self.params.edgenet_dim)

        elif d_name == 'SimplifiedDiscriminatorWithLabel':
            dim_group = [self.params.d_conv_dim, self.params.aux_dim, self.params.lin_dim]
            len_a = len(self.atom_decoder)
            D = SimplifiedDiscriminatorWithLabel(dim_group, len_a, len(self.bond_decoder),
                                                 self.params.dropout,
                                                 gc_activation=get_activation_function(self.params.d_gc_activation),
                                                 dense_activation=get_activation_function(
                                                     self.params.d_dense_activation),
                                                 global_in=self.params.label_length)
        else:
            raise ValueError('invalid discriminator name')

        return G, D

    def get_optimizers(self):
        valid_opt = False
        if self.params.optim_name == 'adam':
            g_optimizer = torch.optim.Adam(self.G.parameters(),
                                           self.params.g_lr, [self.params.beta1, self.params.beta2])
            d_optimizer = torch.optim.Adam(self.D.parameters(), self.params.d_lr,
                                           [self.params.beta1, self.params.beta2])
            valid_opt = True
        if not valid_opt:
            raise ValueError('invalid optimizer name')

        return g_optimizer, d_optimizer

    @classmethod
    def load_from_file(cls, filename):
        class_dict = torch.load(filename)
        new_trainer = cls(class_dict['params'], class_dict['device'])
        new_trainer.load_state_dicts(class_dict)
        new_trainer.step_info = class_dict['step_info']
        new_trainer.epochs_trained = class_dict['epochs_trained']
        new_trainer.device = class_dict['device']
        new_trainer.eval_history = class_dict['eval_history']
        new_trainer.times = class_dict['times']

        return new_trainer

    def load_state_dicts(self, state_dicts):
        for state_dict in state_dicts:
            if state_dict == 'G':
                self.G.load_state_dict(state_dicts[state_dict])
            if state_dict == 'D':
                self.D.load_state_dict(state_dicts[state_dict])
            if state_dict == 'd_optimizer':
                self.d_optimizer.load_state_dict(state_dicts[state_dict])
            if state_dict == 'g_optimizer':
                self.g_optimizer.load_state_dict(state_dicts[state_dict])

    def save_model(self, model_dir):
        info = vars(self)
        saved_info = {}
        for var in info:
            if var == 'G':
                saved_info['G'] = self.G.state_dict()
            elif var == 'D':
                saved_info['D'] = self.D.state_dict()
            elif var == 'd_optimizer':
                saved_info['d_optimizer'] = self.d_optimizer.state_dict()
            elif var == 'g_optimizer':
                saved_info['g_optimizer'] = self.g_optimizer.state_dict()
            else:
                saved_info[var] = info[var]
        filepath = os.path.join('models', model_dir)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            self.params.save(os.path.join(filepath, 'model_params.json'))
        model_path = os.path.join(filepath, 'model_info.pt')
        torch.save(saved_info, model_path)
        print('model saved to {}'.format(model_path))

    def update_eval_history(self, eval_dict):

        for key in eval_dict:
            self.eval_history.setdefault(key, []).append(eval_dict[key])

    @abstractmethod
    def compute_D_loss(self):
        pass

    @abstractmethod
    def compute_G_loss(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @staticmethod
    def matrices2mol(atom_decoder, bond_decoder, node_labels, edge_labels, strict=False):
        """

                function yanked form original MolGAN
        """
        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(atom_decoder[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), bond_decoder[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol


class Basic_GAN_Trainer(GAN_Trainer):

    def __init__(self, model_params, device):

        super(Basic_GAN_Trainer, self).__init__(model_params, device)

    def compute_D_loss(self, a_tensor, x_tensor, z, logger=None):
        logits_real, features_real = self.D(a_tensor, x_tensor)
        d_loss_real = - torch.mean(logits_real)

        # Compute loss with fake images.
        edges_logits, nodes_logits = self.G(z)
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        logits_fake, features_fake = self.D(edges_hat, nodes_hat)
        d_loss_fake = torch.mean(logits_fake)

        # Compute loss for gradient penalty.
        eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
        x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
        x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
        grad_out, meh = self.D(x_int0, x_int1)
        d_loss_gp = self.gradient_penalty(grad_out, x_int0) + self.gradient_penalty(grad_out, x_int1)

        # Backward and optimize.
        d_loss = d_loss_fake + d_loss_real + self.params.lambda_gp * d_loss_gp
        if logger is not None:
            logger['D/loss_real'] = d_loss_real.item()
            logger['D/loss_fake'] = d_loss_fake.item()
            logger['D/loss_gp'] = d_loss_gp.item()
            logger['D/loss_tot'] = d_loss.item()

        return d_loss

    def compute_G_loss(self, z, logger=None):

        self.G.train()
        edges_logits, nodes_logits = self.G(z)
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        logits_fake, features_fake = self.D(edges_hat, nodes_hat)
        g_loss = - torch.mean(logits_fake)

        # Logging.
        if logger is not None:
            logger['G/loss'] = g_loss.item()
        return g_loss

    def evaluate_G(self, z, evaluation_func, draw=False, logger=None):
        self.G.eval()
        edges_logits, nodes_logits = self.G(z)
        (edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.matrices2mol(self.atom_decoder, self.bond_decoder, n_.data.cpu().numpy(), e_.data.cpu().numpy(),
                                  strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        evaluation = None
        if evaluation_func is not None:
            evaluation = evaluation_func(mols)
        if draw:
            for ii in range(10):
                if mols[ii] is not None:
                    Chem.Draw.MolToFile(mols[ii], 'data/mols' + str(self.epochs_trained) + str(ii) + '.png')

        if logger is not None:
            logger['evaluation'] = evaluation

        self.G.train()
        return evaluation

    def train(self, train_loader, batch_size, epochs, evaluation_func=None, model_dir=None, draw=False, logging=False,
              save_every=None, confirmation=False, collapse_check=False):
        """

        Trains the GAN + output molecules after each epoch
        :return:
        """
        start_time = time.time()
        log_dict = None
        for _ in range(epochs):
            for idx, data in enumerate(train_loader):

                if logging:
                    log_dict = {}

                a_tensor = data['edge_adj_mat'].to(self.device)
                x_tensor = data['node_attr'].to(self.device)

                z = get_embedding(batch_size, self.params.z_dim).to(self.device)

                if not (idx % self.params.n_critic == 0):
                    d_loss = self.compute_D_loss(a_tensor, x_tensor, z, log_dict)

                    # backprop
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                if (idx % self.params.n_critic == 0):
                    # Z-to-target

                    g_loss = self.compute_G_loss(z, log_dict)

                    # backprop
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                self.step_info[idx] = log_dict

            self.epochs_trained += 1

            evaluation = self.evaluate_G(z, basic_scores, draw=draw, logger=log_dict)

            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, self.epochs_trained, self.epochs_trained)
            if evaluation:
                for tag, value in evaluation.items():
                    log += ", {}: {:.4f}".format(tag, value)
            print(log)

            if save_every is not None and self.epochs_trained % save_every == 0:
                self.save_model(model_dir)
            # Log update
            '''
            EVAL_func
            m1 = basic_scores(mols)  # 'mols' is output of Fake Reward
            loss.update(m1)
            '''

    def gradient_penalty(self, y, x):

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


class Sim_GAN_Trainer(GAN_Trainer):
    hard_gumbel_sim_metrics = []
    soft_gumbel_sim_metrics = ['kl_div', 'js_div']
    softmax_sim_metrics = ['eig_sim_l1', 'eig_sim_l2', 'kl_div', 'js_div']

    def __init__(self, model_params, device):

        super(Sim_GAN_Trainer, self).__init__(model_params, device)

        if self.params.post_method == 'softmax' and \
                self.params.sim_metric not in Sim_GAN_Trainer.softmax_sim_metrics:
            raise ValueError('not a valid similarity metric with softmax processing')
        elif self.params.post_method == 'hard_gumbel' and \
                self.params.sim_metric not in Sim_GAN_Trainer.hard_gumbel_sim_metrics:
            raise ValueError('not a valid similarity metric with hard gumbel processing')
        elif self.params.post_method == 'soft_gumbel' and \
                self.params.sim_metric not in Sim_GAN_Trainer.soft_gumbel_sim_metrics:
            raise ValueError('not a valid similarity metric with soft gumbel processing')

    def compute_D_loss(self, a_tensor, x_tensor, z, logger=None):
        logits_real, features_real = self.D(a_tensor, x_tensor)
        d_loss_real = - torch.mean(logits_real)

        # Compute loss with fake images.
        edges_logits, nodes_logits = self.G(z)
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        logits_fake, features_fake = self.D(edges_hat, nodes_hat)
        d_loss_fake = torch.mean(logits_fake)

        # Compute loss for gradient penalty.
        eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
        x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
        x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
        grad_out, meh = self.D(x_int0, x_int1)
        d_loss_gp = self.gradient_penalty(grad_out, x_int0) + self.gradient_penalty(grad_out, x_int1)

        # Backward and optimize.
        d_loss = d_loss_fake + d_loss_real + self.params.lambda_gp * d_loss_gp
        if logger is not None:
            logger['D/loss_real'] = d_loss_real.item()
            logger['D/loss_fake'] = d_loss_fake.item()
            logger['D/loss_gp'] = d_loss_gp.item()
            logger['D/loss_tot'] = d_loss.item()

        return d_loss

    def compute_G_loss(self, z, logger=None):

        self.G.train()
        edges_logits, nodes_logits = self.G(z)
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        logits_fake, features_fake = self.D(edges_hat, nodes_hat)
        sim_regularizer = pairwise_batch_comparison(edges_hat, nodes_hat, self.params.sim_metric,
                                                    processed=True, sample=self.params.sim_sample_size,
                                                    reduction='mean')
        adveserial_loss = - torch.mean(logits_fake)
        g_loss = adveserial_loss - self.params.sim_lambda * sim_regularizer

        # Logging.
        if logger is not None:
            logger['G/loss'] = g_loss.item()
            logger['G/sim_reg'] = sim_regularizer.item()
            logger['G/adv_loss'] = adveserial_loss.item()
        return g_loss

    def evaluate_G(self, z, evaluation_func, draw=False, logger=None):
        self.G.eval()
        edges_logits, nodes_logits = self.G(z)
        (edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.matrices2mol(self.atom_decoder, self.bond_decoder, n_.data.cpu().numpy(), e_.data.cpu().numpy(),
                                  strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        evaluation = None
        if evaluation_func is not None:
            evaluation = evaluation_func(mols)
        if draw:
            for ii in range(10):
                if mols[ii] is not None:
                    Chem.Draw.MolToFile(mols[ii], 'data/mols' + str(self.epochs_trained) + str(ii) + '.png')
        # if logger is not None:
        #    logger['evaluation'] = evaluation

        self.G.train()
        return evaluation

    def train(self, train_loader, batch_size, epochs, evaluation_func=None, model_dir=None, draw=False, logging=False,
              save_every=None, confirmation=False, collapse_check=False):
        """

        Trains the GAN + output molecules after each epoch
        :return:
        """
        start_time = time.time()
        log_dict = None
        steps = 0
        for _ in range(epochs):
            for idx, data in enumerate(train_loader):
                if logging:
                    log_dict = {}

                a_tensor = data['edge_adj_mat'].to(self.device)
                x_tensor = data['node_attr'].to(self.device)
                z = get_embedding(batch_size, self.params.z_dim).to(self.device)

                if not (idx % self.params.n_critic == 0):
                    d_loss = self.compute_D_loss(a_tensor, x_tensor, z, log_dict)

                    # backprop
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                if (idx % self.params.n_critic == 0):
                    # Z-to-target

                    g_loss = self.compute_G_loss(z, log_dict)

                    # backprop
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                self.step_info[steps] = log_dict
                steps = steps + 1

            self.epochs_trained += 1

            evaluation = self.evaluate_G(z, basic_scores, draw=draw, logger=log_dict)

            self.update_eval_history(evaluation)

            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, self.epochs_trained, self.epochs_trained)
            if evaluation:
                for tag, value in evaluation.items():
                    log += ", {}: {:.4f}".format(tag, value)
            print(log)

            if save_every is not None and self.epochs_trained % save_every == 0:
                self.save_model(model_dir)
            # Log update
            '''
            EVAL_func
            m1 = basic_scores(mols)  # 'mols' is output of Fake Reward
            loss.update(m1)
            '''
        if confirmation:
            print('TRAINFINISHED')

    def gradient_penalty(self, y, x):

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


class CGAN_Trainer(GAN_Trainer):

    def __init__(self, model_params, device):

        super(CGAN_Trainer, self).__init__(model_params, device)

    def compute_D_loss(self, a_tensor, x_tensor, z, labels, logger=None, labels_added=False):

        if not labels_added:
            z = torch.cat([z, labels], dim=1)
        logits_real, features_real = self.D(a_tensor, x_tensor, labels)
        d_loss_real = - torch.mean(logits_real)

        # Compute loss with fake images.
        edges_logits, nodes_logits = self.G(z)
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        logits_fake, features_fake = self.D(edges_hat, nodes_hat, labels)
        d_loss_fake = torch.mean(logits_fake)

        # Compute loss for gradient penalty.
        eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
        x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
        x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
        # labels = labels.requires_grad_(True)
        grad_out, meh = self.D(x_int0, x_int1, labels)
        d_loss_gp = self.gradient_penalty(grad_out, x_int0) + self.gradient_penalty(grad_out, x_int1)

        # Backward and optimize.
        d_loss = d_loss_fake + d_loss_real + self.params.lambda_gp * d_loss_gp
        if logger is not None:
            logger['D/loss_real'] = d_loss_real.item()
            logger['D/loss_fake'] = d_loss_fake.item()
            logger['D/loss_gp'] = d_loss_gp.item()
            logger['D/loss_tot'] = d_loss.item()

        return d_loss

    def compute_G_loss(self, z, labels, logger=None, labels_added=False):

        self.G.train()
        if not labels_added:
            z = torch.cat([z, labels], dim=1)
        edges_logits, nodes_logits = self.G(z)
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        logits_fake, features_fake = self.D(edges_hat, nodes_hat, labels)
        adveserial_loss = - torch.mean(logits_fake)
        g_loss = adveserial_loss

        # Logging.
        if logger is not None:
            logger['G/loss'] = g_loss.item()
            logger['G/adv_loss'] = adveserial_loss.item()
        return g_loss

    def evaluate_G(self, z, evaluation_func, draw=False, logger=None):
        self.G.eval()
        evaluations = {}
        for labels in self.params.eval_labels:
            label = torch.tensor(labels).unsqueeze(0).repeat(z.size(0), 1).to(self.device)
            eval_z = torch.cat([z, label], dim=1)
            edges_logits, nodes_logits = self.G(eval_z)
            (edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits), self.params.post_method)
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            mols = [
                self.matrices2mol(self.atom_decoder, self.bond_decoder, n_.data.cpu().numpy(), e_.data.cpu().numpy(),
                                  strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
            evaluation = None
            label_tag = f" (label:{str(labels)})"
            if evaluation_func is not None:
                evaluation = evaluation_func(mols)
                evaluation = {k + label_tag: v for k, v in evaluation.items()}
                evaluations.update(evaluation)
            if draw:
                for ii in range(10):
                    if mols[ii] is not None:
                        Chem.Draw.MolToFile(mols[ii],
                                            'data/mols' + str(self.epochs_trained) + str(labels) + str(ii) + '.png')
        # if logger is not None:
        #    logger['evaluation'] = evaluation

        self.G.train()
        return evaluations

    def sample_G(self, z, eval_label):
        label = torch.tensor(eval_label).unsqueeze(0).repeat(z.size(0), 1).to(self.device)
        eval_z = torch.cat([z, label], dim=1)
        edges_logits, nodes_logits = self.G(eval_z)
        (edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits), self.params.post_method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        return edges_hard, nodes_hard

    def get_mols(self, edges_hard, nodes_hard):
        mols = [self.matrices2mol(self.atom_decoder, self.bond_decoder, n_.data.cpu().numpy(), e_.data.cpu().numpy(),
                                  strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    @staticmethod
    def draw_mol(mol, filename):
        Chem.Draw.MolToFile(mol, filename)

    def train(self, train_loader, batch_size, epochs, evaluation_func=None, model_dir=None, draw=False, logging=False,
              save_every=None, confirmation=False, collapse_check=False):
        """

        Trains the GAN + output molecules after each epoch
        :return:
        """
        start_time = time.time()
        log_dict = None
        steps = 0
        collapse_count = 0
        for _ in range(epochs):

            for idx, data in enumerate(train_loader):
                if logging:
                    log_dict = {}

                labels = data['y'].to(self.device)
                a_tensor = data['edge_adj_mat'].to(self.device)
                x_tensor = data['node_attr'].to(self.device)

                # x_tensor = torch.cat([data['node_attr'], label_tensor.repeat(1,self.params.vertexes).unsqueeze(2)], dim=2).to(self.device)
                # z = torch.cat([get_embedding(batch_size, self.params.z_dim), label_tensor], dim=1).to(self.device)

                z = get_embedding(batch_size, self.params.z_dim).to(self.device)

                # concatenate labels to z
                # z = torch.cat([get_embedding(batch_size, self.params.z_dim).to(self.device), labels], dim=1)

                if not (idx % self.params.n_critic == 0):
                    d_loss = self.compute_D_loss(a_tensor, x_tensor, z, labels, log_dict)

                    # backprop
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                if (idx % self.params.n_critic == 0):
                    # Z-to-target

                    g_loss = self.compute_G_loss(z, labels, log_dict)

                    # backprop
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                self.step_info[steps] = log_dict
                steps = steps + 1

            self.epochs_trained += 1

            evaluation = self.evaluate_G(z, conditional_basic_scores, draw=draw, logger=log_dict)

            self.update_eval_history(evaluation)

            et = time.time() - start_time
            self.times.append(et)
            print("epoch time", et)
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, self.epochs_trained, self.epochs_trained)

            for key in self.eval_history:
                if 'unique' in key:
                    if self.eval_history[key][-1] < .01:
                        collapse_count = collapse_count + 1
                        filepath = os.path.join('models', model_dir, 'model_info.pt')
                        if os.path.exists(filepath) and collapse_check:
                            new_path = os.path.join('models', model_dir, 'pre_collapse.pt')
                            # os.rename(filepath, new_path)
                            collapse_check = False
            if evaluation:
                for tag, value in evaluation.items():
                    print(tag, value)
                    log += ", {}: {:.4f}".format(tag, value)
            print(log)

            if save_every is not None and self.epochs_trained % save_every == 0:
                self.save_model(model_dir)

            if collapse_count == 6:
                print('MODEL COLLAPSE')
                break
            # Log update
            '''
            EVAL_func
            m1 = basic_scores(mols)  # 'mols' is output of Fake Reward
            loss.update(m1)
            '''

    def gradient_penalty(self, y, x):

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


if __name__ == "__main__":
    from datasets import SparseMolDataset
    from torch.utils.data import DataLoader
    from params import SimGANParams, param_set_to_df, CGANParams, BasicGANParams
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model type (options: base, cond, sim)")
    parser.add_argument("prefix", help="prefix to identify given group of experiments. Try to make unique")
    parser.add_argument("index", help="index within param_file to run", type=int)
    parser.add_argument("dataset_name", help="name of dataset file/folder")
    parser.add_argument("--idx_file", help="if sharing train,valid, test index, file containing those")
    parser.add_argument("--num_runs", help="number of times to run model", type=int, default=1)
    parser.add_argument("--save_every", help="how many epochs until model is saved", type=int, default=1)
    parser.add_argument("--collapse_check", help="assess mode collapse", action='store_true', default=False)
    args = parser.parse_args()

    # before this need to have constructed dataset
    df = pd.read_csv(args.prefix + '_master_file.tsv', index_col=False, sep='\t')
    filename = args.prefix + '_master_file.tsv'
    if args.model == 'base':
        param_obj = BasicGANParams.from_df_file(filename, args.index)
        data = SparseMolDataset('data/' + args.dataset_name)
    elif args.model == 'cond':
        param_obj = CGANParams.from_df_file(filename, args.index)
        data = SparseMolDataset('data/' + args.dataset_name, label_map=param_obj.mapping)
    elif args.model == 'sim':
        param_obj = SimGANParams.from_df_file(filename, args.index)
        data = SparseMolDataset('data/' + args.dataset_name)
    else:
        raise ValueError("not a valid GAN model (options: base, cgan, simgan)")

    param_obj.atom_num_types = data.dataset.atom_num_types
    param_obj.bond_num_types = data.dataset.bond_num_types
    param_obj.vertexes = data.adj_size
    param_obj.bond_decoder = data.dataset.bond_decoder_m
    param_obj.atom_decoder = data.dataset.atom_decoder_m
    print(param_obj.dict)
    test = DataLoader(data, batch_size=param_obj.batch_size, drop_last=True, shuffle=True)
    if args.model == 'base':
        trainer = Basic_GAN_Trainer(param_obj, 'cuda')
    elif args.model == 'cond':
        trainer = CGAN_Trainer(param_obj, 'cuda')
    elif args.model == 'sim':
        trainer = Sim_GAN_Trainer(param_obj, 'cuda')
    time_string = time.strftime("%Y%m%d-%H%M%S")
    model_dir = args.prefix + '_' + args.dataset_name.split('.')[0] + str(args.index) + '_' + time_string
    trainer.train(test, param_obj.batch_size, param_obj.epochs, model_dir=model_dir, save_every=args.save_every,
                  logging=True, confirmation=True, collapse_check=args.collapse_check)
