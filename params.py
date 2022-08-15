from abc import ABC, abstractmethod
import json
from itertools import product
import pandas as pd

class Params(ABC):
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, params):
        self.__dict__.update(params)
        self.check_params()
    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
               params = json.load(f)
        return cls(params)

    @classmethod
    def from_dict(cls, params):
        return cls(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update_from_json(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def update_from_dict(self, param_dict):
        """Load parameters from dict"""

        self.__dict__.update(param_dict)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

    @abstractmethod
    def check_params(self):
        pass

class SimGANParams(Params):

    param_names = ['g_conv_dim', 'z_dim', 'vertexes', 'atom_num_types', 'bond_num_types', 'dropout', 'd_conv_dim',
                   'aux_dim', 'lin_dim', 'g_lr', 'd_lr', 'beta1', 'beta2', 'post_method', 'lambda_gp','n_critic',
                   'layer_init', 'generator_type', 'discriminator_type', 'd_gc_activation', 'd_dense_activation',
                   'optim_name', 'sim_metric', 'sim_sample_size', 'sim_lambda']
    df_converter = {'g_conv_dim': eval, 'd_conv_dim': eval, 'lin_dim': eval}
    def __init__(self, params):
        super(SimGANParams, self).__init__(params)

    def check_params(self):
        for param in SimGANParams.param_names:
            if param not in self.__dict__:
                raise ValueError('missing required param: {}'.format(param))

    @classmethod
    def from_df_file(cls, df_file, idx):
        df = pd.read_csv(df_file, index_col=None, sep='\t',converters=cls.df_converter)
        return cls(df.to_dict(orient='records')[idx])

    @classmethod
    def generate_param_set(cls, **kwargs):
        param_values = [v for k, v in kwargs.items()]
        keys = [k for k, v in kwargs.items()]
        param_sets = product(*param_values)
        param_sets = [cls({k: v for k, v in zip(keys, param_set)}) for param_set in param_sets]
        return param_sets

class CGANParams(Params):

    param_names = ['g_conv_dim', 'z_dim', 'vertexes', 'atom_num_types', 'bond_num_types', 'dropout', 'd_conv_dim',
                   'aux_dim', 'lin_dim', 'g_lr', 'd_lr', 'beta1', 'beta2', 'post_method', 'lambda_gp','n_critic',
                   'layer_init', 'generator_type', 'discriminator_type', 'd_gc_activation', 'd_dense_activation',
                   'optim_name', 'label_length', 'eval_labels', 'mapping']
    df_converter = {'g_conv_dim': eval, 'd_conv_dim': eval, 'lin_dim': eval, 'eval_labels': eval, 'mapping': eval}

    def __init__(self, params):
        super(CGANParams, self).__init__(params)

    def check_params(self):
        for param in CGANParams.param_names:
            if param not in self.__dict__:
                raise ValueError('missing required param: {}'.format(param))

    @classmethod
    def from_df_file(cls, df_file, idx):
        df = pd.read_csv(df_file, index_col=None, sep='\t',converters=cls.df_converter)
        return cls(df.to_dict(orient='records')[idx])

    @classmethod
    def generate_param_set(cls, **kwargs):
        param_values = [v for k, v in kwargs.items()]
        keys = [k for k, v in kwargs.items()]
        param_sets = product(*param_values)
        param_sets = [cls({k: v for k, v in zip(keys, param_set)}) for param_set in param_sets]
        return param_sets


class BasicGANParams(Params):
    param_names = ['g_conv_dim', 'z_dim', 'vertexes', 'atom_num_types', 'bond_num_types', 'dropout', 'd_conv_dim',
                   'aux_dim', 'lin_dim', 'g_lr', 'd_lr', 'beta1', 'beta2', 'post_method', 'lambda_gp', 'n_critic',
                   'layer_init', 'generator_type', 'discriminator_type', 'd_gc_activation', 'd_dense_activation',
                   'optim_name']
    df_converter = {'g_conv_dim': eval, 'd_conv_dim': eval, 'lin_dim': eval}

    def __init__(self, params):
        super(BasicGANParams, self).__init__(params)

    def check_params(self):
        for param in BasicGANParams.param_names:
            if param not in self.__dict__:
                raise ValueError('missing required param: {}'.format(param))

    @classmethod
    def from_df_file(cls, df_file, idx):
        df = pd.read_csv(df_file, index_col=None, sep='\t', converters=cls.df_converter)
        return cls(df.to_dict(orient='records')[idx])

    @classmethod
    def generate_param_set(cls, **kwargs):
        param_values = [v for k, v in kwargs.items()]
        keys = [k for k, v in kwargs.items()]
        param_sets = product(*param_values)
        param_sets = [cls({k: v for k, v in zip(keys, param_set)}) for param_set in param_sets]
        return param_sets

def param_set_to_df(param_set):

    return pd.DataFrame([vars(params) for params in param_set])

if __name__ == "__main__":
    param_dict = {}
    param_dict['g_conv_dim'] = [[128,256,512]]
    param_dict['batch_size'] = [128]
    param_dict['epochs'] = [50]
    param_dict['z_dim'] = [32, 40]
    param_dict['vertexes'] = [9]
    param_dict['atom_num_types'] = [5]
    param_dict['bond_num_types'] = [5]
    param_dict['dropout'] = [0]
    param_dict['d_conv_dim'] = [[128,64],[128,128,64]]
    param_dict['aux_dim'] = [128]
    param_dict['lin_dim'] = [[128, 64]]
    param_dict['g_lr'] = [.0001, .00001, .001]
    param_dict['d_lr'] = [.001, .0001, .00001, .00005]
    param_dict['beta1'] = [.9]
    param_dict['beta2'] = [.999]
    param_dict['post_method'] = ['softmax']
    param_dict['lambda_gp'] = [5]
    param_dict['n_critic'] = [5,10]
    param_dict['layer_init'] = ['golort']
    param_dict['generator_type'] = ['GraphGeneratorConditional']
    param_dict['discriminator_type'] = ['SimplifiedDiscriminatorWithLabel']
    param_dict['d_gc_activation'] = ['ftanh', 'frelu']
    param_dict['d_dense_activation'] = ['tanh', 'relu']
    param_dict['optim_name'] = ['adam']
    param_dict['label_length'] = [1]
    param_dict['eval_labels'] = [[0.]]
    param_dict['mapping'] = [{}]
    #param_dict['sim_metric'] = ["eig_sim_l2"]
    #param_dict['sim_sample_size'] = [50]
    #param_dict['sim_lambda'] = [0, 1, 10]
    param_set = CGANParams.generate_param_set(**param_dict)
    param_df = param_set_to_df(param_set)
    print(param_df.dtypes)
    param_df.to_csv('CGAN_zinc_master_file.tsv', sep='\t', index=False)
    param_df.to_csv('CGAN_zinc_index.csv', columns=[], header=False)