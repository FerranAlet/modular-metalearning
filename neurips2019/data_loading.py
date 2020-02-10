import numpy as np
import copy

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import h5py
import random


"""
Creates the datasets.

If you want to create your own way of creating datasets;
subclass Dataset and MetaDataset
"""

# Default parameters
SEED = 2
EPS = 1e-9
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
# torch.initial_seed(SEED)

###########
# DATASET #
###########
class Dataset(object):
  def __init__(self, train, val, test, name=None):
    self.train = train
    self.val = val
    self.test = test
    self.name = name

###################
# DATASET CREATOR #
###################


class MetaDataset(data.Dataset):

  def __getitem__(self, idx):
    return self.ALL[idx]#.data

  def __len__(self):
    return len(self.ALL)

  def __init__(self,
               train, val, test,
               limit_data=-1, smaller_MVals=[], max_datasets=1e9,
               train_val_are_all=False,
               filename=None, normalize_output=False, normalize=False,
               mtrain=None, mval=None, mtest=None,
               RS=None, split_by_file=None):

    #Random state
    if RS is None: self.RS = np.random.RandomState(seed=SEED)
    elif type(RS) == type(1): self.RS = np.random.RandomState(seed=RS)
    else: self.RS = RS
    # proportion of datasets in each split
    self.mtrain= mtrain
    self.mval = mval
    self.mtest = mtest
    # proportion of data in each split
    self.train = train
    self.val = val
    self.test = test

    # whether to make train data == val data and exclude test
    self.train_val_are_all = train_val_are_all

    # Other customizations
    self.max_datasets = max_datasets
    self.limit_data = limit_data
    self.normalize = normalize
    self.normalize_output = normalize_output
    self.smaller_MVals = smaller_MVals
    self.split_by_file = split_by_file

    # Create data
    self.create_datasets(filename)
    assert self.MTRAIN is not None and self.MVAL is not None and self.MTEST is not None and self.ALL is not None

    if self.normalize:
      self.apply_normalization()
    elif self.normalize_output:
      self.apply_output_normalization()

    if self.smaller_MVals != []:
      assert max(self.smaller_MVals) < self.MVAL[0].train, (
        'all smaller_MVals should be  smaller than train' +
        str(max(self.smaller_MVals)) + ' vs ' + str(self.MVAL[0].train))
      self.prev_mval = self.mval
      for dataset in self.MVAL:
        if dataset.name is not None: dataset.name += '_' + str(dataset.train)
      for train_sz in self.smaller_MVals:
        self.add_smaller_cases(self.prev_mval, train_sz)
      self.mval = len(self.MVAL)

    if self.MTRAIN[0].name is not None:
      self.MTRAIN.sort(key=lambda x: x.name)
      self.MVAL.sort(key=lambda x: x.name)
      self.MTEST.sort(key=lambda x: x.name)

  def add_smaller_cases(self, old_idx, new_train):
    '''
    Creates new cases with smaller training and appends them to self.MVAL
    '''
    for i in range(old_idx):
      aux = copy.deepcopy(self.MVAL[i])
      name = aux.name.split('_')
      aux.train = new_train
      aux.TrainInput = aux.TrainInput[:aux.train]
      aux.TrainOutput = aux.TrainOutput[:aux.train]
      aux.name = '_'.join(name[:-1] + [str(aux.train)])
      self.ALL.append(aux)
      self.MVAL.append(self.ALL[-1])

  def apply_output_normalization(self):
    '''
    Applies normalization to all inputs and all outputs in MetaDataset
    at the same time
    '''
    # Check all inputs and all outputs have the same size
    out_shape = self.MTRAIN[0].TrainOutput.shape[1]
    for dataset in self.ALL:
      assert dataset.TestOutput.shape[1] == out_shape
    # Compute mean and std
    ALL_OUT = np.concatenate([dataset.ValOutput for dataset in self.MVAL], 0)
    self.output_mean = np.mean(ALL_OUT, 0)
    self.output_std = np.maximum(EPS, np.std(ALL_OUT, 0))
    print('MVAL Output mean: ', self.output_mean)
    print('MVAL Output std: ', self.output_std)

    ALL_OUT = np.concatenate([dataset.ValOutput for dataset in self.MTRAIN], 0)
    self.output_mean = np.mean(ALL_OUT, 0)
    self.output_std = np.maximum(EPS, np.std(ALL_OUT, 0))
    print('Output mean: ', self.output_mean)
    print('Output std: ', self.output_std)

    for dataset in self.ALL:
      # Copy unnormalized versions
      dataset.UTrainOutput = copy.deepcopy(dataset.TrainOutput)
      dataset.UValOutput = copy.deepcopy(dataset.ValOutput)

      dataset.TrainOutput = (dataset.TrainOutput - self.output_mean) / self.output_std
      dataset.ValOutput = (dataset.ValOutput - self.output_mean) / self.output_std
      dataset.TestOutput = (dataset.TestOutput - self.output_mean) / self.output_std

  def apply_normalization(self):
    '''
    Applies normalization to all inputs and all outputs in MetaDataset
    at the same time
    '''
    # Check all inputs and all outputs have the same size
    inp_shape = self.MTRAIN[0].TrainInput.shape[1]
    out_shape = self.MTRAIN[0].TrainOutput.shape[1]
    for dataset in self.ALL:
      assert dataset.TrainInput.shape[1] == inp_shape
      assert dataset.TestOutput.shape[1] == out_shape
    # Compute mean and std
    ALL_IN = np.concatenate([dataset.ValInput for dataset in self.MVAL], 0)
    ALL_OUT = np.concatenate([dataset.ValOutput for dataset in self.MVAL], 0)
    self.input_mean = np.mean(ALL_IN, 0)
    self.input_std = np.maximum(EPS, np.std(ALL_IN, 0))
    self.output_mean = np.mean(ALL_OUT, 0)
    self.output_std = np.maximum(EPS, np.std(ALL_OUT, 0))
    print('MVAL Input mean: ', self.input_mean)
    print('MVAL Input std: ', self.input_std)
    print('MVAL Output mean: ', self.output_mean)
    print('MVAL Output std: ', self.output_std)

    ALL_IN = np.concatenate([dataset.ValInput for dataset in self.MTRAIN], 0)
    ALL_OUT = np.concatenate([dataset.ValOutput for dataset in self.MTRAIN], 0)
    self.input_mean = np.mean(ALL_IN, 0)
    self.input_std = np.maximum(EPS, np.std(ALL_IN, 0))
    self.output_mean = np.mean(ALL_OUT, 0)
    self.output_std = np.maximum(EPS, np.std(ALL_OUT, 0))
    print('Input mean: ', self.input_mean)
    print('Input std: ', self.input_std)
    print('Output mean: ', self.output_mean)
    print('Output std: ', self.output_std)

    for dataset in self.ALL:
      # Copy unnormalized versions
      dataset.UTrainInput = copy.deepcopy(dataset.TrainInput)
      dataset.UTrainOutput = copy.deepcopy(dataset.TrainOutput)
      dataset.UValInput = copy.deepcopy(dataset.ValInput)
      dataset.UValOutput = copy.deepcopy(dataset.ValOutput)

      dataset.TrainInput = (dataset.TrainInput - self.input_mean) / self.input_std
      dataset.TrainOutput = (dataset.TrainOutput - self.output_mean) / self.output_std
      dataset.ValInput = (dataset.ValInput - self.input_mean) / self.input_std
      dataset.ValOutput = (dataset.ValOutput - self.output_mean) / self.output_std
      dataset.TestInput = (dataset.TestInput - self.input_mean) / self.input_std
      dataset.TestOutput = (dataset.TestOutput - self.output_mean) / self.output_std

  def normalize_input(self, x):
    return (x - self.input_mean) / self.input_std

  def denormalize_input(self, x):
    return self.input_mean + self.input_std * x

  def normalize_output(self, x):
    return (x - self.output_mean) / self.output_std

  def denormalize_output(self, x):
    return self.output_mean + self.output_std * x

  def create_datasets(self, filename):
    # Populates self.ALL, self.MTRAIN, self.MVAL, self.MTEST
    raise NotImplementedError ("please subclass MetaDataset")


class NPDataset(Dataset):
  def __init__(self,
               input_data, output_data,
               train, val, test,
               train_val_are_all,
               name=None, limit=-1,
               edge_data=None, structure_idx=None):
    # input_data, output_data are np arrays
    super(NPDataset, self).__init__(train=train, val=val, test=test, name=name)

    # Handle input/output with more than 2 dims,
    # compress to 2, but remember.
    self.original_input_shape = input_data.shape
    self.original_output_shape = output_data.shape

    # structure associated with this dataset
    self.structure_idx = structure_idx

    assert input_data.shape[0] == output_data.shape[0]
    if limit >= 0:
      input_data = input_data[:limit, :]
      output_data = output_data[:limit, :]
    len_data = input_data.shape[0]
    if abs(train + val + test - 1.) < 1e-6:
      self.train = int(round(len_data * train))
      self.val = int(round(len_data * val))
      self.test = len_data - self.train - self.val
      assert self.train + self.val + self.test == len_data
      assert min([self.train, self.val, self.test]) >= 0

    if train_val_are_all:
      self.train = len_data
      self.val = len_data
      self.test = 0
      self.TrainInput = input_data
      self.TrainOutput = output_data
      self.ValInput = input_data
      self.ValOutput = output_data
      self.TestInput = input_data[:0]
      self.TestOutput = output_data[:0]
    else:
      self.TrainInput = input_data[:self.train]
      self.TrainOutput = output_data[:self.train]
      self.ValInput = input_data[self.train:self.train + self.val]
      self.ValOutput = output_data[self.train:self.train + self.val]
      self.TestInput = input_data[self.train + self.val:]
      self.TestOutput = output_data[self.train + self.val:]

    self.Edges = edge_data

    if np.array_equal(self.TrainInput, self.TrainOutput):
      # self.TrainInput = self.TrainInput[:-1]
      self.TrainOutput = self.TrainOutput[1:]
      # self.ValInput = self.ValInput[:-1]
      self.ValOutput = self.ValOutput[1:]
      # self.TestInput = self.TestInput[:-1]
      self.TestOutput = self.TestOutput[1:]
      self.train = self.TrainInput.shape[0]
      self.val = self.ValInput.shape[0]
      self.test =  self.TestInput.shape[0]
    self.Train = [self.TrainInput, self.TrainOutput]
    self.Val = [self.ValInput, self.ValOutput]
    self.Test = [self.TestInput, self.TestOutput]

    self.All = [input_data, output_data, edge_data]


class MetaNpySelfRegressDataset(MetaDataset):
  '''
  Loads 3 [Train,Val,Test] .npy files with arrays of shape
  [cases, inp_dim_1, inp_dim_2, inp_dim_k]
  '''

  # Inherits __init__, __len__, __getitem__

  def _create_meta_dataset(self, array, edges, names, train_val_are_all=False, hard_max_datasets=1e10):
    '''
    From an array of shape [n,...] creates many datasets
    '''
    list_to_append = []
    max_datasets = min(self.max_datasets, hard_max_datasets)
    for i in range(min(array.shape[0], max_datasets)):
      list_to_append.append(NPDataset(
        input_data=array[i], output_data=array[i], edge_data=edges[i],
        structure_idx=i, #limit=48,
        train=self.train, val=self.val, test=self.test,
        train_val_are_all=train_val_are_all,
        name=names[i]))
    return list_to_append

  def create_datasets(self, filename):
    '''
    Populates self.ALL, self.MTRAIN, self.MVAL, self.MTEST
    '''
    # (num_datasets, num_nodes, num_traj_steps, input_dims)
    MTrainArray = np.load('data/state_' + filename + '_train.npy')
    MValArray = np.load('data/state_' + filename + '_val.npy')
    MTestArray = np.load('data/state_' + filename + '_test.npy')

    # import ipdb; ipdb.set_trace()

    # (num datasets, num connections (minus diagonals)
    MTrainEdges = np.load('data/edges_' + filename + '_train.npy')
    MValEdges = np.load('data/edges_' + filename + '_val.npy')
    MTestEdges = np.load('data/edges_' + filename + '_test.npy')

    # get rid of self-edges from Edges labels
    def delete_single_diagonal(A):
      ''' deletes A's diagonal (removing self-edges)'''
      return np.delete(A, range(0, A.shape[0] ** 2, (A.shape[0] + 1))).reshape(A.shape[0], (A.shape[1] - 1))

    def delete_self_edges(A):
      ''' makes A into a list of square matrices representing the graph's adjacency
      matrix of edge module types, then makes each square matrix into a n x (n-1) matrix
      by deleting its diagonal (removing self-edges), then reshapes them to original shape[0]'''
      if len(A.shape) == 2:
        adj_mat = A.reshape((A.shape[0], np.sqrt(A.shape[1]).astype(int), -1))
      else:
        adj_mat = A
      split_squeezed = map(lambda x: x.squeeze(0), np.split(adj_mat, adj_mat.shape[0]))
      return np.stack(list(map(delete_single_diagonal, split_squeezed))).reshape(A.shape[0], -1)


    for edges_data in [MTrainEdges, MValEdges, MTestEdges]:
      edges_data[edges_data == -1] = 0,


    MTrainNames = ['train_' + str(i) for i in range(MTrainArray.shape[0])]
    MValNames = ['val_' + str(i) for i in range(MValArray.shape[0])]
    MTestNames = ['test_' + str(i) for i in range(MTestArray.shape[0])]

    self.MTRAIN = self._create_meta_dataset(MTrainArray, MTrainEdges, MTrainNames,
            train_val_are_all=True)
    self.MTEST = self._create_meta_dataset(MValArray, MValEdges, MValNames,
            hard_max_datasets=max(250, self.max_datasets // 4))
    self.MVAL = self._create_meta_dataset(MTestArray, MTestEdges, MTestNames,
            hard_max_datasets=self.max_datasets)
    self.ALL = self.MTRAIN + self.MVAL + self.MTEST

    self.mtrain = len(self.MTRAIN)
    self.mtest = len(self.MVAL)
    self.mval = len(self.MTEST)

class MetaHDFDataset(MetaDataset):
  # Inherits __init__, __len__, __getitem__

  def create_datasets(self, filename):
    '''
    Populates self.ALL, self.MTRAIN, self.MVAL, self.MTEST
    '''
    #Read HDF5
    f = h5py.File(filename, 'r')
    DATA = [[f[key], f[key.replace('IN','OUT')]]
        for key in sorted(list(f.keys())) if 'IN' in key]
    NAMES = [name.replace('-IN','')
        for name in sorted(list(f.keys())) if 'IN' in name]
    print('Names = ', NAMES[:10])
    print('Num datasets = ', len(NAMES))

    #DATA contains a list of 'object'=datasets
    self.mval_fraction, self.mtest_fraction = self.mval, self.mtest
    if self.max_datasets < len(DATA):
      #Subsample DATA randomly
      subsampled_idx  = self.RS.choice(len(DATA), size=self.max_datasets,
          replace=False)
      DATA = [d for (i_d,d) in enumerate(DATA) if i_d in subsampled_idx]
      NAMES = [d for (i_d,d) in enumerate(NAMES) if i_d in subsampled_idx]

    self.limit_data = min(self.limit_data, DATA[0][0][()].shape[0])
    tot_datasets = min(DATA[0][0][()].shape[0] // self.limit_data,
        self.max_datasets//len(DATA))*len(DATA)
    self.mval = int(round(self.mval*tot_datasets))
    self.mtest = int(round(self.mtest*tot_datasets))
    self.mtrain = int(tot_datasets - self.mtest - self.mval)
    assert self.mtrain > 0
    self.ALL = []
    num_datasets_hist = []

    self.MTRAIN = []
    self.MVAL = []
    self.MTEST = []
    num_datasets_hist = []
    shuffled_pairs = list(zip(NAMES, DATA))
    random.shuffle(shuffled_pairs)
    for (name, data) in shuffled_pairs:
      input_data = data[0][()]
      output_data = data[1][()]
      num_datasets = min(input_data.shape[0] // self.limit_data,
          self.max_datasets//len(DATA))
      num_datasets_hist.append(num_datasets)
      if num_datasets == 0:
        import pdb; pdb.set_trace()
        print(name+ ' has no datasets') ; continue
      if False: #self.shuffle:
        #Shuffle in unison
        rng_state = self.RS.get_state()
        self.RS.shuffle(input_data)
        self.RS.set_state(rng_state)
        self.RS.shuffle(output_data)
      list_to_append = self.ALL #same for all datasets created from this file
      if self.split_by_file:
        if len(self.MTRAIN) < self.mtrain: list_to_append = self.MTRAIN
        elif len(self.MVAL) < self.mval: list_to_append = self.MVAL
        elif len(self.MTEST) < self.mtest: list_to_append = self.MTEST
        else: assert False, 'something doesnt add up'
      # print(num_datasets)
      for k in range(num_datasets):
        list_to_append.append(NPDataset(
          input_data=input_data[k*self.limit_data:(k+1)*self.limit_data],
          output_data=output_data[k*self.limit_data:(k+1)*self.limit_data],
          train_val_are_all=False,
          name=name,train=self.train, val=self.val, test=self.test,
          structure_idx=k))
    if self.split_by_file:
      self.ALL = self.MTRAIN + self.MVAL + self.MTEST
    else:
      self.RS.shuffle(self.ALL)
      [self.MTRAIN, self.MTEST, self.MVAL] = [
          self.ALL[:self.mtrain],
          self.ALL[self.mtrain:self.mtrain+self.mtest],
          self.ALL[self.mtrain+self.mtest:]]
    print('Goal meta sizes: ', self.mtrain, self.mval, self.mtest)
    (self.mtrain, self.mval, self.mtest) = (
        len(self.MTRAIN), len(self.MVAL), len(self.MTEST))
    print('Final meta sizes: ', self.mtrain, self.mval, self.mtest)
    return self.ALL



def convert_to_torch_tensors(np_dataset, nn_device):
  torch_dataset = copy.deepcopy(np_dataset)
  for i, dataset in enumerate(torch_dataset.ALL):
    torch_dataset.ALL[i].TrainInput = (
      torch.from_numpy(dataset.TrainInput).float().to(nn_device))
    torch_dataset.ALL[i].TrainOutput = (
      torch.from_numpy(dataset.TrainOutput).float().to(nn_device))
    torch_dataset.ALL[i].ValInput = (
      torch.from_numpy(dataset.ValInput).float().to(nn_device))
    torch_dataset.ALL[i].ValOutput = (
      torch.from_numpy(dataset.ValOutput).float().to(nn_device))
    torch_dataset.ALL[i].TestInput = (
      torch.from_numpy(dataset.TestInput).float().to(nn_device))
    torch_dataset.ALL[i].TestOutput = (
      torch.from_numpy(dataset.TestOutput).float().to(nn_device))
  return torch_dataset


def get_data_loaders(dataset, batch_size=32, shuffle=True):

  # Creating data indices for training and validation splits:
  dataset_size = len(dataset)
  indices = list(range(dataset_size))

  train_indices, val_indices, test_indices \
    = indices[:dataset.mtrain], indices[dataset.mtrain: dataset.mtrain + dataset.mval], \
      indices[dataset.mtrain + dataset.mval:]

  train_dataset = Subset(dataset, train_indices)
  val_dataset = Subset(dataset, val_indices)
  test_dataset = Subset(dataset, test_indices)

  def collate(batch):
    return batch

  custom_collate = collate

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate) if len(test_dataset.indices) else None

  return train_loader, val_loader, test_loader


if __name__ == '__main__':
  dataset = MetaNpySelfRegressDataset("data/feat_charged5")
