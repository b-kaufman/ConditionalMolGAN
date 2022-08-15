import pickle
import numpy as np
import utils
import argparse

from rdkit import Chem
'''
if __name__ == '__main__':
    from progress_bar import ProgressBar
else:
    from utils.progress_bar import ProgressBar
'''
from datetime import datetime


class SparseMolecularDataset():

    def load(self, filename, subset=1):

        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_new_idx_info(self, filename, subset=1):

        with open(filename, 'rb') as f:
            idx_dict = pickle.load(f)

        self.train_idx = idx_dict['train_idx']
        self.validation_idx = idx_dict['validation_idx']
        self.test_idx = idx_dict['test_idx']

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save_idx_info(self, filename):
        idx_dict = {'train_idx': self.train_idx, 'validation_idx': self.validation_idx, 'test_idx': self.test_idx}

        with open(filename, 'wb') as f:
            pickle.dump(idx_dict, f)

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))

        if filename.endswith('.sdf'):
            # list of rdkit molecules
            self.data = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
        elif filename.endswith('.smi'):
            # list of rdkit molecules
            self.data = [Chem.MolFromSmiles(line) for line in open(filename, 'r').readlines()]

        # modify data based on parameters
        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(len(self.data),
                                                                              len(Chem.SDMolSupplier(filename)),
                                                                              '' if add_h else 'not '))

        self._generate_encoders_decoders()
        self._generate_AX()

        # it contains the all the molecules stored as rdkit.Chem objects
        self.data = np.array(self.data)

        # a (N, 9, 9) tensor where N is the length of the dataset and each 9x9 matrix contains the 
        # indices of the positions of the ones in the one-hot representation of the adjacency tensor
        # (see self._genA)
        self.data_A = np.stack(self.data_A)

        # a (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the 
        # indices of the positions of the ones in the one-hot representation of the annotation matrix
        # (see self._genX)
        self.data_X = np.stack(self.data_X)

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        # dictionaries. encoder maps atomic_num to index. decoder is index to atomic_num
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        # self.mw_decoder_m = {i: Chem.rdchem.Atom(l).GetMass() for i, l in enumerate(atom_labels)}

        self.atom_num_types = len(atom_labels)
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
                                                                    for mol in self.data
                                                                    for bond in mol.GetBonds())))
        # dictionaries. encoder maps bond_type to index. decoder is index to bond_type
        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))

    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')
        # pr = ProgressBar(60, len(self.data))

        data = []
        data_A = []
        data_X = []
        data_LogPNorm = []
        data_Donors = []
        data_Acceptors = []
        data_rBonds = []
        data_MW = []

        max_length = max(mol.GetNumAtoms() for mol in self.data)
        print('Max molecule length is:', max_length)
        #build representation for each molecule
        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            if A is not None:
                data.append(mol)
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))

                data_LogPNorm.append(utils.MolecularMetrics.water_octanol_partition_coefficient_scores([mol]))
                data_Donors.append(utils.MolecularMetrics.HDonors([mol]))
                data_Acceptors.append(utils.MolecularMetrics.HAcceptors([mol]))
                data_MW.append(utils.MolecularMetrics.molWt([mol]))
                data_rBonds.append(utils.MolecularMetrics.rotatableBonds([mol]))

                if i % 5000 == 0:
                    print(i, 'complete')

        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                           len(self.data)))

        self.data = data
        self.data_A = data_A
        self.data_X = data_X
        self.label_data = {'logp' : data_LogPNorm, 'donors' : data_Donors, 'acceptors' : data_Acceptors,
                           'mw' : data_MW, 'rotate' : data_rBonds}
        self.label_data = {k: np.stack(v) for k, v in self.label_data.items()}

        self.__len = len(self.data)

    def _genA(self, mol, connected=True, max_length=None):

        # generates adjacency matrix indexed based on idx provided
        # by rdkit mol object. adj mat is 2d and encodes bond types with differing labels (single 1, double 2 etc.)
        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        # number of single bonds going into each atom.
        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)

        # return if all nodes are connected
        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        # vector of indices for each atom type in molecule padded by 0s to max mol length

        max_length = max_length if max_length is not None else mol.GetNumAtoms()


        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                    max_length - mol.GetNumAtoms()), dtype=np.int32)

    def _genS(self, mol, max_length=None):

        # vector of indices for each char in SMILES string padded by 0s to max mol length

        max_length = max_length if max_length is not None else len(Chem.MolToSmiles(mol))

        return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
                    max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32)

    def _genF(self, mol, max_length=None):

        # list comprehension generates 2d array NxF where N is atoms and F is features
        # this is stacked on top of padding to the max_length

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array([[*[a.GetDegree() == i for i in range(5)],
                              *[a.GetExplicitValence() == i for i in range(9)],
                              *[int(a.GetHybridization()) == i for i in range(1, 7)],
                              *[a.GetImplicitValence() == i for i in range(9)],
                              a.GetIsAromatic(),
                              a.GetNoImplicit(),
                              *[a.GetNumExplicitHs() == i for i in range(5)],
                              *[a.GetNumImplicitHs() == i for i in range(5)],
                              *[a.GetNumRadicalElectrons() == i for i in range(5)],
                              a.IsInRing(),
                              *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def matrices2mol(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def matrices2MW(self, node_labels):
        mw = 0.

        for node_label in node_labels:
            mw += self.mw_decoder_m[node_label]
        return mw


    def seq2mol(self, seq, strict=False):
        mol = Chem.MolFromSmiles(''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def one_hot(self, nparray, depth=0, on_value=1, off_value=0):
        if depth == 0:
            depth = np.max(nparray) + 1
        assert np.max(nparray) < depth, "the max index of nparray: {} is larger than depth: {}".format(np.max(nparray),
                                                                                                       depth)
        shape = nparray.shape
        out = np.ones((*shape, depth)) * off_value
        indices = []
        for i in range(nparray.ndim):
            tiles = [1] * nparray.ndim
            s = [1] * nparray.ndim
            s[i] = -1
            r = np.arange(shape[i]).reshape(s)
            if i > 0:
                tiles[i - 1] = shape[i - 1]
                r = np.tile(r, tiles)
            indices.append(r)
        indices.append(nparray)
        out[tuple(indices)] = on_value
        return out

    def _transform_labels(self, transform_map):
        for k in transform_map:
            if transform_map[k] == 'disc':

                self.label_data[k] = self.discretize(self.label_data[k])

            if transform_map[k] == 'norm':

                self.label_data[k] = self.normalize(self.label_data[k])

            if 'bin' in transform_map[k]:
                bins = int(transform_map[k].split('_')[1])
                self.label_data[k] = self.bin(self.label_data[k], bins)

    def normalize(self, dat):
        d_max = np.amax(dat)
        d_min = np.amin(dat)
        return (dat-d_min)/(d_max - d_min)

    def discretize(self, dat):
        avg = np.mean(dat)
        res = np.where(dat <= avg, 0., dat)
        res = np.where(res > avg, 1., res)
        return res

    def bin(self, dat, bins):
        bin_size = 1./float(bins)
        bins = np.asarray([np.quantile(dat, bin_size * x + bin_size) for x in range(bins)])
        last = 0
        for idx, bin in enumerate(bins):
            dat = np.where(np.logical_and(dat > last, dat <= bin), float(idx), dat)
            last = bin
        return dat

    def _generate_train_validation_test(self, validation, test):

        self.log('Creating train, validation and test sets..')

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train:train + validation]
        self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
            train, validation, test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [obj[idx[counter:counter + batch_size]]
                      for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                  self.data_D, self.data_F, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                           self.data_D, self.data_F, self.data_Le, self.data_Lv)]

        return [counter] + output

    def next_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]

        return out[1:]

    def next_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]

        return out[1:]

    def next_test_batch(self, batch_size=None):
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]

        return out[1:]

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="path to sdf or smi file to process into sparse dataset", default='data/gdb9.sdf')
    parser.add_argument("--sparse_dataset_file", help="output filepath", default ='data/gdb9_9nodes.sparsedataset')
    parser.add_argument("--mol_size_limit", help="max number of atoms allowed", type=int, default =9)
    args = parser.parse_args()

    data = SparseMolecularDataset()
    data.generate(args.input_file, filters=lambda x: x.GetNumAtoms() <= args.mol_size_limit)
    data.save(args.sparse_dataset_file)

    #data = SparseMolecularDataset()
    #data.generate('data/zinc_data.smi', filters=lambda x: x.GetNumAtoms() <= 20)
    #data.save('data/zinc250_20nodes.sparsedataset')

    # data = SparseMolecularDataset()
    # data.generate('data/qm9_5k.smi', validation=0.00021, test=0.00021)  # , filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('data/qm9_5k.sparsedataset')
