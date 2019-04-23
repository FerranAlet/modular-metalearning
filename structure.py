import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm as Tqdm
from torch.nn import ParameterList

class Structure():
  def __init__(self, args):
    #Assert module specifications are consistent.
    if not hasattr(self, 'type_modules'): self.type_modules = []
    if self.type_modules != []:
      pass
    elif type(args.type_modules) == type(''):
      self.type_modules = args.type_modules.split(',')
    else:
      assert type(args.type_modules) == type([])
      assert type(args.type_modules[0]) == type('a')
      self.type_modules = args.type_modules
    if type(args.num_modules) == type(''):
      self.num_modules = list(map(int,
        args.num_modules.split(',')))
    else:
      assert type(args.num_modules) == type([])
      assert type(args.num_modules[0]) == type(1)
      self.num_modules = args.num_modules
    self.num_types = len(self.num_modules)
    assert len(self.type_modules) == self.num_types, (str(self.type_modules) + \
        ' should have '+str(self.num_types)+' elts.')
    self.tot_modules = sum(self.num_modules)

    self.usage_normalization = 1e-9
    self.has_global_variable = False
    self.StructureParameters = ParameterList()

  def initialize_structure(self):
    '''
    Create a random structure
    '''
    raise NotImplementedError

  def propose_new_structure(self, new_structure):
    '''
    Given a structure, new_structure, make a modification
    '''
    raise NotImplementedError

  def update_structure(self, structure, step=None):
    '''
    Optional function updating some aspects of the structure
    for instance to adapt to some parameters that changed by SGD
    '''
    return

  def update_Usage_counters(self, METRICS, T):
    '''
    Updates table of fraction of times module is used for each dataset.
    '''
    raise NotImplementedError

  def modules_given_structure(self, structure):
    raise NotImplementedError

  ###########################
  ##  Optional functions   ##
  ###########################
  def propose_new_global_structure(self):
    return None

  def save_customized_files(self, directory):
    return

  def get_plot_name(self, args, plot_name):
    '''
    Modifies the plot_name
    '''
    return

  def update_PosUsage_counters(self, METRICS):
    '''
    Updates table of fraction of times module is used for each dataset.
    and for each slot.
    '''
    return

  def update_customized_counters(self, METRICS=None):
    '''
    Updates tables depending on the pertinent structure.
    '''
    return

  def update_customized_stats(self, BG=None):
    '''
    Updates tables depending on the pertinent structure.
    '''
    return

  def plot_customized_usage_rate(self, directory=None):
    return

  #####################################################
  ## Common functions, not expected to be customized ##
  #####################################################

  def plot_usage(self, directory):
    if self.Usage is not None:
      cax = plt.gca().matshow(self.Usage/self.usage_normalization)
      plt.gcf().colorbar(cax)
      plt.savefig(os.path.join(directory, 'usage-rate'))
      plt.clf()

  def initialize_all_structures(self, T, mtrain_copies=1):
    self.TrainStructures = [None for _ in range(mtrain_copies * T.mtrain)]
    self.ValStructures = [None for _ in T.MVAL]
    for i in Tqdm(range(len(self.TrainStructures))):
      self.TrainStructures[i] = self.initialize_structure()
      self.TrainStructures[i]['original_input_shape'] = (
              T.MTRAIN[i%T.mtrain].original_input_shape)
      self.TrainStructures[i]['original_output_shape'] = (
              T.MTRAIN[i%T.mtrain].original_output_shape)
    for i in range(len(self.ValStructures)):
      self.ValStructures[i] = self.initialize_structure()
      self.ValStructures[i]['original_input_shape'] = (
              T.MVAL[i].original_input_shape)
      self.ValStructures[i]['original_output_shape'] = (
              T.MVAL[i].original_output_shape)

  ###########################################################
  ## Default functions                                     ##
  ## For some functions, many structures use the same code ##
  ###########################################################

  def default_update_Usage_counters(self, METRICS, T):
    '''
    Updates table of fraction of times module is used for each dataset.
    '''
    eps = 1e-3
    self.Usage *= (1-eps)
    for i_s, structure in enumerate(self.TrainStructures+self.ValStructures):
      for m in structure['modules']:
        self.Usage[i_s][m] += eps
    names = ([T.MTRAIN[i%T.mtrain].name
        for i in range(len(self.TrainStructures))] + [_.name for _ in T.MVAL])
    names = list(enumerate(names))
    names.sort(key = lambda x : x[1])
    values = self.Usage[[_[0] for _ in names],:]
    METRICS['Usage'] = [[values[i][j] for j in range(values.shape[1])]
        for i in range(values.shape[0]) ]
    METRICS['Usage-names'] = [_[1] for _ in names]

  def default_initialize_structure(self):
    structure = {'modules':[]}
    for i in range(self.structure_size):
      act_type = np.random.randint(self.structure_size)
      act_mod = np.random.randint(self.num_modules[act_type])
      structure['modules'].append(self.Modules[act_type][act_mod])
    return structure

  def default_initialize_fixed_structure(self):
    #Position 'i' in the structure can only be filled by a node of type 'i'
    assert self.num_types == self.structure_size
    structure = {'modules':[]}
    for i in range(self.num_types):
      act_type = i
      act_mod = np.random.randint(self.num_modules[act_type])
      structure['modules'].append(self.Modules[act_type][act_mod])
    return structure

  def default_propose_new_structure(self, new_structure):
    pos = np.random.randint(len(new_structure['modules']))
    act_type = np.random.randint(self.num_types)
    act_mod = np.random.randint(self.num_modules[act_type])
    new_structure['modules'][pos] = self.Modules[act_type][act_mod]

  def default_propose_new_fixed_structure(self, new_structure):
    #Position 'i' in the structure can only be filled by a node of type 'i'
    pos = np.random.randint(len(new_structure['modules']))
    act_type = pos
    act_mod = np.random.randint(self.num_modules[act_type])
    new_structure['modules'][pos] = self.Modules[act_type][act_mod]
