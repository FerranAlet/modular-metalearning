"""
Creates the datasets.

If you want to create your own way of creating datasets;
subclass Dataset and MetaDataset
"""
import copy
import h5py
import numpy as np
import random

#Default parameters
SEED = 2
EPS = 1e-9

###########
# DATASET #
###########
class Dataset(object):
  def __init__(self, train, val, test, RS = None, shuffle = True, name=None):
    #Random state
    if RS is None: self.RS = np.random.RandomState(seed=SEED)
    elif type(RS) == type(1): self.RS = np.random.RandomState(seed=RS)
    else: self.RS = RS
    self.train = train
    self.val = val
    self.test = test
    self.shuffle = shuffle
    self.name = name

###################
# DATASET CREATOR #
###################
class MetaDataset(object):
  def __init__(self,
      train, val, test,  #Size of each dataset
      mtrain, mval, mtest, #Number of metacases
      shuffle = True,
      limit_data=-1,
      split_by_file=False,
      smaller_MVals=[],
      max_datasets=1e9,
      RS = None, filename=None, normalize=False):

    #Random state
    if RS is None: self.RS = np.random.RandomState(seed=SEED)
    elif type(RS) == type(1): self.RS = np.random.RandomState(seed=RS)
    else: self.RS = RS #RandomState
    random.seed(SEED)

    #Parameters for each testcase
    self.train = train
    self.val = val
    self.test = test

    self.mtrain = mtrain #meta-train size
    self.mval = mval #meta-validation size
    self.mtest = mtest #meta-test size

    #Other customizations
    self.max_datasets = max_datasets
    self.shuffle = shuffle
    self.limit_data = limit_data
    self.normalize = normalize
    self.split_by_file = split_by_file
    self.smaller_MVals = smaller_MVals

    #Create data
    self.create_datasets(filename)
    if self.normalize:
      self.apply_normalization()

    if self.smaller_MVals != []:
      assert max(self.smaller_MVals) < self.MVAL[0].train, (
          'all smaller_MVals should be  smaller than train'+
          str(max(self.smaller_MVals))+' vs '+str(self.MVAL[0].train))
      self.prev_mval = self.mval
      for dataset in self.MVAL:
        if dataset.name is not None: dataset.name += '_' + str(dataset.train)
      for train_sz in self.smaller_MVals:
        self.add_smaller_cases(self.prev_mval, train_sz)
      self.mval = len(self.MVAL)

    if self.MTRAIN[0].name is not None:
      self.MTRAIN.sort(key = lambda x : x.name)
      self.MVAL.sort(key = lambda x : x.name)
      self.MTEST.sort(key = lambda x : x.name)

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
    ALL_IN = np.concatenate([dataset.ValInput for dataset in self.MVAL] , 0)
    ALL_OUT = np.concatenate([dataset.ValOutput for dataset in self.MVAL],0)
    self.input_mean = np.mean(ALL_IN, 0)
    self.input_std = np.maximum(EPS, np.std(ALL_IN, 0))
    self.output_mean = np.mean(ALL_OUT, 0)
    self.output_std = np.maximum(EPS, np.std(ALL_OUT,0))
    print('MVAL Input mean: ', self.input_mean)
    print('MVAL Input std: ', self.input_std)
    print('MVAL Output mean: ', self.output_mean)
    print('MVAL Output std: ', self.output_std)

    ALL_IN = np.concatenate([dataset.ValInput for dataset in self.MTRAIN] , 0)
    ALL_OUT = np.concatenate([dataset.ValOutput for dataset in self.MTRAIN],0)
    self.input_mean = np.mean(ALL_IN, 0)
    self.input_std = np.maximum(EPS, np.std(ALL_IN, 0))
    self.output_mean = np.mean(ALL_OUT, 0)
    self.output_std = np.maximum(EPS, np.std(ALL_OUT,0))
    print('Input mean: ', self.input_mean)
    print('Input std: ', self.input_std)
    print('Output mean: ', self.output_mean)
    print('Output std: ', self.output_std)

    for dataset in self.ALL:
      #Copy unnormalized versions
      dataset.UTrainInput = copy.deepcopy(dataset.TrainInput)
      dataset.UTrainOutput = copy.deepcopy(dataset.TrainOutput)
      dataset.UValInput = copy.deepcopy(dataset.ValInput)
      dataset.UValOutput = copy.deepcopy(dataset.ValOutput)

      dataset.TrainInput = (dataset.TrainInput-self.input_mean)/self.input_std
      dataset.TrainOutput = (dataset.TrainOutput-
          self.output_mean)/self.output_std
      dataset.ValInput = (dataset.ValInput-self.input_mean)/self.input_std
      dataset.ValOutput = (dataset.ValOutput-self.output_mean)/self.output_std
      dataset.TestInput = (dataset.TestInput-self.input_mean)/self.input_std
      dataset.TestOutput = (dataset.TestOutput-self.output_mean)/self.output_std

  def normalize_input(self, x):
    return (x-self.input_mean) / self.input_std

  def denormalize_input(self, x):
    return self.input_mean + self.input_std*x

  def normalize_output(self, x):
    return (x-self.output_mean) / self.output_std

  def denormalize_output(self, x):
    return self.output_mean + self.output_std*x

  def create_datasets(self, filename=None):
    #Populates self.ALL, self.MTRAIN, self.MVAL, self.MTEST
    raise NotImplementedError

######################
# Datasets from hdf5 #
######################
class NPDataset(Dataset):
  def __init__(self,
      input_data, output_data,
      train, val, test,
      name=None, limit=-1,
      RS = None, shuffle = True,
      train_val_are_all=False):
    #input_data, output_data are np arrays
    super(NPDataset, self).__init__(train=train, val=val, test=test,
        RS = RS, shuffle = shuffle, name=name)
    #Handle input/output with more than 2 dims,
    #compress to 2, but remember.
    self.original_input_shape = input_data.shape
    self.original_output_shape = output_data.shape
    input_data = input_data.reshape(input_data.shape[0],-1)
    output_data = output_data.reshape(output_data.shape[0],-1)

    assert input_data.shape[0] == output_data.shape[0]
    if limit >=0:
      input_data = input_data[:limit,:]
      output_data = output_data[:limit,:]
    len_data = input_data.shape[0]
    if abs(train+val+test-1.) < 1e-6:
      self.train = int(round(len_data*train))
      self.val = int(round(len_data*val))
      self.test = len_data - self.train-self.val
      assert self.train + self.val + self.test == len_data
      assert min([self.train, self.val, self.test]) >=0

    if self.shuffle: #Shuffle in unison
      rng_state = self.RS.get_state()
      self.RS.shuffle(input_data)
      self.RS.set_state(rng_state)
      self.RS.shuffle(output_data)

    self.All = [input_data, output_data]

    if train_val_are_all:
      self.train = len_data
      self.val = len_data
      self.test = 0
      self.TrainInput =  input_data
      self.TrainOutput = output_data
      self.ValInput =    input_data
      self.ValOutput =   output_data
      self.TestInput =   input_data [:0]
      self.TestOutput =  output_data[:0]
    else:
      self.TrainInput =  input_data [:self.train]
      self.TrainOutput = output_data[:self.train]
      self.ValInput =    input_data [self.train:self.train+self.val]
      self.ValOutput =   output_data[self.train:self.train+self.val]
      self.TestInput =   input_data [self.train+self.val:]
      self.TestOutput =  output_data[self.train+self.val:]

    self.Train = [self.TrainInput, self.TrainOutput]
    self.Val = [self.ValInput, self.ValOutput]
    self.Test = [self.TestInput, self.TestOutput]


class MetaHDFDataset(MetaDataset):
  # Inherits __init__

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
        print(name+ ' has no datasets') ; continue
      if self.shuffle:
        #Shuffle in unison
        rng_state = self.RS.get_state()
        self.RS.shuffle(input_data)
        self.RS.set_state(rng_state)
        self.RS.shuffle(output_data)
      else: input('Are you sure you dont want to shuffle?')
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
          name=name,train=self.train, val=self.val, test=self.test))
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

class MetaNpySelfRegressDataset(MetaDataset):
  '''
  Loads 3[Train,Val,Test] .npy files with arrays of shape
  [cases,inp_dim_1, inp_dim_2,inp_dim_k]
  '''
  # Inherits __init__

  def create_meta_dataset(self, array, names, hard_max_datasets=1e10):
    '''
    From an array of shape [n,...] creates many datasets
    '''
    list_to_append = []
    max_datasets = min(self.max_datasets, hard_max_datasets)
    for i in range(min(array.shape[0], max_datasets)):
      list_to_append.append(NPDataset(
        input_data=array[i],
        output_data=array[i],
        name=names[i], train=self.train, val=self.val, test=self.test,
        shuffle=False, #dont shuffle
        train_val_are_all=True))
    return list_to_append

  def create_datasets(self, filename):
    '''
    Populates self.ALL, self.MTRAIN, self.MVAL, self.MTEST
    '''
    #Read .npy
    MTrainArray = np.load(filename+'_train.npy')
    MValArray = np.load(filename+'_val.npy')
    MTestArray = np.load(filename+'_test.npy')

    MTrainArray = np.transpose(MTrainArray, [0,2,1,3])
    MValArray = np.transpose(MValArray, [0,2,1,3])
    MTestArray = np.transpose(MTestArray, [0,2,1,3])
    import pdb; pdb.set_trace()
    MTrainNames = ['train_'+str(i) for i in range(MTrainArray.shape[0])]
    MValNames = ['val_'+str(i) for i in range(MValArray.shape[0])]
    MTestNames = ['test_'+str(i) for i in range(MTestArray.shape[0])]

    self.MTRAIN = self.create_meta_dataset(MTrainArray, MTrainNames)
    self.MVAL = self.create_meta_dataset(MValArray, MValNames,
        hard_max_datasets=self.max_datasets//4)
    self.MTEST = self.create_meta_dataset(MTestArray, MTestNames)
    self.mtrain = len(self.MTRAIN)
    self.mval = len(self.MVAL)
    self.mtest = len(self.MTEST)
    self.ALL = self.MTRAIN + self.MVAL + self.MTEST
    return self.ALL

if __name__ == '__main__':
  mod = MetaHDFDataset(filename='data/sumed_noduplicates_1000.hdf5')
  print(mod.ALL)
  print(mod.MTRAIN, type(mod.MTRAIN), len(mod.MTRAIN))
  print(type(mod.MTRAIN[0].TrainInput), mod.MTRAIN[0].TrainInput.shape)
  print(type(mod.MTRAIN[0].TrainOutput), mod.MTRAIN[0].TrainOutput.shape)
