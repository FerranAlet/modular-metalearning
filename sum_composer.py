'''
Subclass for the composition 'sum'
'''
from __future__ import print_function
import torch
from composition import Composer
from structure import Structure
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  nn_device='cuda:0'
else:
  torch.set_default_tensor_type('torch.FloatTensor')
  nn_device="cpu"
torch.device(nn_device)

class Sum_Composer(Composer):
  def __init__(self, composer, module_list, loss_fn=None, structure={},
      instructions={}):
    super(Sum_Composer, self).__init__(composer=composer,
        module_list=module_list, loss_fn=loss_fn,
        structure=structure, instructions=instructions)

  def forward_no_weights(self, x):
    res = []
    for mod in self.structure['modules']: res.append(self.module_list[mod](x))
    x = torch.sum(torch.stack(res), 0)
    return x

  def forward_with_weights(self, x, weights):
    res = []
    for mod in self.structure['modules']:
      res.append(self.module_list[mod](x,
        weights=weights, prefix='module_list.'+str(mod)+'.features.'))
    return torch.sum(torch.stack(res), 0)

class Sum_Structure(Structure):
  def __init__(self, args):
    self.composer = 'sum'
    self.composer_class = Sum_Composer
    self.composer_abbreviation = 'SUM'
    self.structure_size = args.structure_size
    super(Sum_Structure, self).__init__(args=args)

  def propose_new_structure(self, new_structure):
    return self.default_propose_new_structure(new_structure)

  def initialize_structure(self):
    return self.default_initialize_structure()

  def update_Usage_counters(self, METRICS, T):
    return self.default_update_Usage_counters(METRICS, T)

  def modules_given_structure(self, structure):
    return structure['modules']
