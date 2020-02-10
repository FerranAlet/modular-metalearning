'''
Composes multiple modules into a single network.

Some parts are inspired by:
https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py
'''
from __future__ import print_function
import torch
from torch import nn
torch.set_default_tensor_type('torch.cuda.FloatTensor')
nn_device='cuda:0'
torch.device(nn_device)

#####################################################################
#                                                                   #
# If you create a new composition grammar you have to               #
# create a (name)_composer.py with functions:                       #
#                                                                   #
# - forward_no_weights                                              #
# - forward_with_weighs: NOT NEEDED IF YOU DON'T USE MAML           #
#     same as with weights but manually forward looping             #
#     through the modules                                           #
# We suggest looking at currently coded grammars as references.     #
# - sum_composer is the simplest                                    #
# - GNN_composer is the most complex                                #
#                                                                   #
#####################################################################

class Composer(nn.Module):
  '''
  Composes multiple modules into a single net.
  '''
  def __init__(self, composer, module_list, loss_fn=None, structure={},
          instructions={}):
    '''
    composer: string describing the composer type
    structure: specifies how to compose which modules
    loss_fn: loss function
    instructions: can be left blank, customizable non-computation parameters
    '''
    super(Composer, self).__init__()
    self.module_list = module_list
    assert type(module_list) == type(torch.nn.ModuleList())
    self.num_modules = len(self.module_list)
    self.composer = composer
    self.loss_fn = loss_fn
    self.structure = structure
    self.instructions = instructions

  def forward_no_weights(self, x):
    """
    Specifies how to get the final result given structure and input
    Method must be implemented in subclass.
    """
    raise NotImplementedError

  def forward_with_weights(self, x, weights):
    """
    Similar to forward_no_weights, but fixing weights
    Method must be implemented in subclass.
    """
    raise NotImplementedError

  def forward(self, x, weights=None):
    ##  TODO: use the forward method of each method for weights!=None
    ##  instead of manual code.
    if weights is None:
      return self.forward_no_weights(x)
    else:
      return self.forward_with_weights(x, weights)

  def net_forward(self, x, weights=None):
    return self.forward(x, weights)

  def copy_weights(self, net):
    '''Set this module's weights to be the same as those of 'net' '''
    raise NotImplementedError
    for m_from, m_to in zip(net.modules(), self.modules()):
      if (isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d)
          or isinstance(m_to, nn.BatchNorm2d)):
        m_to.weight.data = m_from.weight.data.clone()
        if m_to.bias is not None:
            m_to.bias.data = m_from.bias.data.clone()
