'''
Performing the inner loop of MAML.

Heavily inspired by:
https://github.com/katerakelly/pytorch-maml/blob/master/src/inner_loop.py
'''
from collections import OrderedDict
import torch
import torch.nn as nn
from custom_module import torch_NN
torch.set_default_tensor_type('torch.cuda.FloatTensor')
nn_device='cuda:0'
torch.device(nn_device)


class InnerLoop():
  '''
  This module performs the inner loop of MAML
  The forward method updates weights with gradient steps on training data,
  then computes and returns a meta-gradient w.r.t. validation data.
  '''
  def __init__(self, baseComposer, module_list, loss_fn,
      num_updates, step_size):
    # super(InnerLoop, self).__init__(composer=baseComposer.composer,
    #     module_list=module_list, loss_fn=loss_fn)
    #Composer, already initialized
    self.loss_fn = loss_fn
    self.C = baseComposer
    self.C.cuda()
    #Number of updates to be taken
    self.num_updates = num_updates
    #Step size for the updates
    self.step_size = step_size
    self.meta_batch_size = 1

  def net_forward(self, x, weights=None):
    return self.C.forward(x, weights)

  def forward(self, dataset):
    # # Test net before training, should be random accuracy
    # tr_pre_loss, __ = self.evaluate(dataset.TrainInput, dataset.TrainOutput)
    # val_pre_loss, __ = self.evaluate(dataset.ValInput, dataset.ValOutput)
    fast_weights = OrderedDict((name, param) for (name, param)
        in self.C.named_parameters())
    for i in range(self.num_updates):
      in_, target = dataset.TrainInput, dataset.TrainOutput
      if i==0:
        loss, _ = self.forward_pass(in_, target)
        grads = torch.autograd.grad(loss, self.C.parameters(),
            create_graph=True, allow_unused=True)
      else:
        loss, _ = self.forward_pass(in_, target, fast_weights)
        grads = torch.autograd.grad(loss, fast_weights.values(),
            create_graph=True, allow_unused=True)
      fast_weights = OrderedDict((name, param - self.step_size*grad)
          for ((name, param), grad) in zip(fast_weights.items(), grads)
          if grad is not None)
    # Test net after training, should be better than random
    tr_post_loss, train_ans = self.evaluate(dataset.TrainInput,
        dataset.TrainOutput, weights=fast_weights)
    val_post_loss, val_ans = self.evaluate(dataset.ValInput, dataset.ValOutput,
        weights=fast_weights)
    # import pdb; pdb.set_trace()

    # Compute the meta gradient and return it
    in_, target = dataset.ValInput, dataset.ValOutput
    loss = val_post_loss/self.meta_batch_size
    grads = torch.autograd.grad(loss,
        self.C.parameters(), allow_unused=True)
    meta_grads = {name: g for ((name, _), g)
        in zip(self.C.named_parameters(), grads)}
    metrics = (tr_post_loss, val_post_loss, train_ans, val_ans)
    return metrics, meta_grads

  def forward_pass(self, in_, target, weights=None):
    ''' Run data through net, return loss and output '''
    input_var = torch.autograd.Variable(in_).cuda(non_blocking=True)
    target_var = torch.autograd.Variable(target).cuda(non_blocking=True)
    # Run the batch through the net, compute loss
    out = self.net_forward(input_var, weights)
    loss = self.loss_fn(out, target_var)
    return loss, out

  def evaluate(self, inp, out, weights=None):
    '''evaluate the net on (inp, out)'''
    #Simpler than pytorch-maml/src/score.py bc I don't have acc nor batches
    return self.forward_pass(inp, out, weights)

  def copy_weights(self, net):
    '''Set this module's weights to be the same as those of 'net' '''
    for (i_m, (m_from, m_to)) in enumerate(
        zip(net.module_list, self.C.module_list)):
      assert not(isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d)
          or isinstance(m_to, nn.BatchNorm2d))
      assert isinstance(m_to, torch_NN)
      self.C.module_list[i_m].copy_weights(net.module_list[i_m])
