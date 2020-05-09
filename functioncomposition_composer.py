'''
Subclass for the composition 'composition'
'''

from __future__ import print_function
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
nn_device='cuda:0'
torch.device(nn_device)
from composition import Composer
from structure import Structure

class FunctionComposition_Composer(Composer):
  def __init__(self, composer, module_list, loss_fn=None, structure={},
      instructions={}):
    super(FunctionComposition_Composer, self).__init__(composer=composer,
        module_list=module_list, loss_fn=loss_fn,
        structure=structure, instructions=instructions)

  def forward_no_weights(self, x):
    for mod in self.structure['modules']:
      x = self.module_list[mod](x)
    return x

  def forward_with_weights(self, x, weights):
    for mod in self.structure['modules']:
      x = self.module_list[mod](x,
        weights=weights, prefix='module_list.'+str(mod)+'.features.')
    return x

class FunctionComposition_Structure(Structure):
  def __init__(self, args):
    self.composer = 'functionCompostion'
    self.composer_class = FunctionComposition_Composer
    self.composer_abbreviation = 'C'
    self.structure_size = args.structure_size
    super(FunctionComposition_Structure, self).__init__(args=args)

  def propose_new_structure(self, new_structure):
    return self.default_propose_new_structure(new_structure)

  def initialize_structure(self):
    return self.default_initialize_structure()

  def update_Usage_counters(self, METRICS, T):
    return self.default_update_Usage_counters(METRICS, T)

  def modules_given_structure(self, structure):
    return structure['modules']
