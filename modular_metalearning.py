from pylab import rcParams
rcParams['figure.figsize'] = 15, 15
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')
import copy
import numpy as np
from maml_inner_loop import InnerLoop
from custom_module import torch_NN
from dataset import MetaHDFDataset, MetaNpySelfRegressDataset
from tqdm import tqdm as Tqdm
import os
import shutil


class BounceGrad(object):
  def __init__(self, S, args, tensorboardX_writer=None):
    ##############################
    # Parse parameters from args #
    ##############################
    self.S = S
    self.composer = args.composer
    #MAML parameters
    self.MAML = args.MAML
    self.MAML_loss_fn = (lambda x,y : torch.mean((x-y)**2))
    self.MAML_inner_updates = args.MAML_inner_updates
    self.MAML_step_size = args.MAML_step_size

    # Device
    if not torch.cuda.is_available() or args.nn_device == 'cpu':
      self.nn_device = 'cpu'
      torch.set_default_tensor_type('torch.FloatTensor')
    else:
      self.nn_device = args.nn_device
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.device(self.nn_device)
    torch.manual_seed(args.torch_seed)

    #Other parameters
    self.mtrain_copies = args.mtrain_copies
    self.normalize_data = args.normalize_data
    self.do_bounce = args.do_bounce
    self.meta_lr = args.meta_lr
    self.initial_temp = args.initial_temp
    self.initial_acc = args.initial_acc
    self.execute_gd_every = args.meta_batch_size
    self.smaller_MVals = (list(map(int, args.smaller_MVals.split(',')))
        if args.smaller_MVals!='' else [])
    self.split_by_file = args.split_by_file
    self.limit_data = args.limit_data
    self.max_datasets = args.max_datasets
    self.data_split = [x/100 for x in map(int, args.data_split.split(','))]
    self.meta_split = [x/100 for x in map(int, args.meta_split.split(','))]

    self.load_modules = args.load_modules
    self.load_structures_and_metrics = args.load_structures_and_metrics
    self.save_modules = args.save_modules
    if len(self.save_modules)>0 and self.save_modules[-1] != '/':
      self.save_modules += '/' #make it a folder

    #Create dataset and modules
    self.data_desc = args.data_desc
    self.create_dataset()
    self.create_modules()
    self.optimization_steps = args.optimization_steps #num. of BOUNCE-GRAD steps

    #Automatic plot_name
    self.get_plot_name(args)
    if self.save_modules == '':
      self.save_modules = self.plot_name
      print('Saving modules to ', self.plot_name)
    if os.path.exists(self.plot_name): shutil.rmtree(self.plot_name)
    os.makedirs(self.plot_name)
    self.store_video = args.store_video
    if self.store_video:
      os.makedirs(self.plot_name+'/video')
    #Metrics
    self.writer = tensorboardX_writer
    self.METRICS = {}
    keys_that_are_empty_lists = ['NumberToWords']
    for k in keys_that_are_empty_lists: self.METRICS[k] = []
    self.METRICS['WordsToNumber'] = {}
    self.METRICS['Sharing'] = [[0. for _ in range(50)] for __ in range(50)]
    self.last_comb = None
    self.last_comb_eval = None
    self.current = None
    self.current_Mtrain = None
    self.current_Meval = None
    self.perm_sample_modules = None
    self.perm_sample_fns = None
    self.plot_ymax = args.plot_ymax
    self.plot_freq = args.plot_freq

    #Structure params
    self.S.Usage = np.zeros((self.mtrain_copies*self.T.mtrain+self.T.mval, len(self.L)))
    self.S.save_customized_files(directory=self.plot_name)

  def get_plot_name(self, args):
    shorter_data_desc = self.data_desc.split('/')[-1].split('.')[0]
    if len(shorter_data_desc) <4: shorter_data_desc = self.data_desc[-4:]
    self.plot_name = shorter_data_desc + '_'
    self.plot_name += self.S.composer_abbreviation + '_'
    if self.MAML:
      if max(self.S.num_modules)>1: self.plot_name += 'MOMA'
      else: self.plot_name += 'MAML'
    else:
      if max(self.S.num_modules)>1: self.plot_name += 'SA'
      else: self.plot_name += 'BIGNET'
    self.plot_name += '_' + str(self.limit_data) + '_' + str(self.max_datasets)
    nn_size = 0
    for net in self.S.type_modules: #dumb hash of module sizes
      aux = net.split('-')
      for num in aux:
        try: int_num = int(num)
        except: int_num = 0
        nn_size += int_num
    self.plot_name += '_NNS' + str(nn_size)
    if self.split_by_file:
      self.plot_name += '_split'
    else: self.plot_name += '_file=shuffled'
    self.plot_name += '_steps=' + str(self.optimization_steps)
    self.plot_name += '_lr=' + str(self.meta_lr)
    self.plot_name += '_Mupdt=' + str(self.MAML_inner_updates)
    self.plot_name += '_copies='+str(self.mtrain_copies)
    print('plot_name: ', self.plot_name)
    if args.plot_name.startswith('overwrite-'):
      self.plot_name = '-'.join(args.plot_name.split('-')[1:])
      print('Args overwrote plot name to:', self.plot_name)
    elif args.plot_name == 'dummy':
      ans = input('Want to change plot_name to dummy?')
      if ans.lower() in ['y', 'yes']:
        print('Changed to dummy')
        self.plot_name = 'dummy'
      else: print('Keeping automatic name')
    elif args.plot_name == 'default':
      input('I will not keep your name, see above')
    else: self.plot_name = args.plot_name + '__' + self.plot_name
    self.S.get_plot_name(args, self.plot_name)
    if self.plot_name[-1] != '/': self.plot_name += '/'

  def create_modules(self):
    '''
    Creates modules following num_modules and type_modules
    '''
    self.L = nn.ModuleList() #Library of PyTorch Modules
    self.S.Modules = []
    self.nn_inp = []
    self.nn_out = []
    self.nn_hid = []
    self.nn_act = []
    for (t, (num, typ)) in enumerate(zip(self.S.num_modules, self.S.type_modules)):
      l = []
      #'final_act'-#inp-#hid1-#hid2-...-#out
      #example: 'affine-123-64-64-42'
      typ_split = typ.split('-')
      for i in range(len(typ_split)):
        try:
          typ_split[i] = int(typ_split[i])
        except: pass
      self.nn_act.append(typ_split[0])
      self.nn_inp.append(typ_split[1])
      self.nn_hid.append(typ_split[2:-1])
      self.nn_out.append(typ_split[-1])
      for _ in range(num):
        aux_nn = torch_NN(inp=self.nn_inp[t], out=self.nn_out[t],
          hidden=self.nn_hid[t],
          final_act=self.nn_act[t]).to(device=self.nn_device)
        l.append(len(self.L))
        self.L.append(aux_nn)
      self.S.Modules.append(l)
    if self.load_modules != '': self.load_L(self.load_modules)
    self.SOpt = torch.optim.Adam(self.L.parameters(), lr=self.meta_lr)
    self.SOpt_scheduler = ReduceLROnPlateau(
        optimizer=self.SOpt, factor=1/2.,
        mode='min', patience=50, threshold=0, cooldown=300, verbose=True,
        min_lr=1e-4)
    self.LocalOpt =None #optimizes custom parameters --> uses Train, not Val
    self.initial_norms = [torch.norm(_) for m in self.L for _ in m.parameters()]

  def create_dataset(self):
    '''
    Creates dataset by calling dataset.py and transforms it to pytorch tensors.
    '''
    # Parsing
    self.parsed_data_desc = self.data_desc.split('-')
    self.data_source = self.parsed_data_desc[0]
    if self.data_source.startswith('HDF5'):
      self.D = MetaHDFDataset(
          mtrain=self.meta_split[0], mval=self.meta_split[1],
          mtest=self.meta_split[2],
          train=self.data_split[0], val=self.data_split[1],
          test=self.data_split[2],
          filename=self.data_source.split('@')[1],
          limit_data=self.limit_data,
          max_datasets=self.max_datasets,
          split_by_file=self.split_by_file,
          smaller_MVals=self.smaller_MVals,
          normalize=self.normalize_data)
      self.function_depth = 2
    elif self.data_source.startswith('NPY_SR'):
      self.D = MetaNpySelfRegressDataset(
          mtrain=self.meta_split[0], mval=self.meta_split[1],
          mtest=self.meta_split[2],
          train=self.data_split[0], val=self.data_split[1],
          test=self.data_split[2],
          filename=self.data_source.split('@')[1],
          limit_data=self.limit_data,
          max_datasets=self.max_datasets,
          split_by_file=self.split_by_file,
          smaller_MVals=self.smaller_MVals,
          normalize=self.normalize_data)
    else: assert False, self.data_source + ' hasnt been implemented yet'
    self.D_mtrain = len(self.D.MTRAIN)
    self.D_mval = len(self.D.MVAL)
    self.D_mtest = len(self.D.MTEST)
    # Convert to pytorch tensors
    self.T = copy.deepcopy(self.D)
    for i, dataset in enumerate(self.T.ALL):
      self.T.ALL[i].TrainInput = (
              torch.from_numpy(dataset.TrainInput).float().to(self.nn_device))
      self.T.ALL[i].TrainOutput = (
              torch.from_numpy(dataset.TrainOutput).float().to(self.nn_device))
      self.T.ALL[i].ValInput = (
              torch.from_numpy(dataset.ValInput).float().to(self.nn_device))
      self.T.ALL[i].ValOutput = (
              torch.from_numpy(dataset.ValOutput).float().to(self.nn_device))
      self.T.ALL[i].TestInput = (
              torch.from_numpy(dataset.TestInput).float().to(self.nn_device))
      self.T.ALL[i].TestOutput = (
              torch.from_numpy(dataset.TestOutput).float().to(self.nn_device))

    # Create running answers  = search ensembles
    # They weren't used in the paper, but may be in the future,
    # as ensembles generally perform better than any single structure.
    self.answers_running_score = 1e-5
    self.ans_eps = 1e-1 #3e-2
    self.MTrainAnswers = [[None, None]
        for _ in range(self.mtrain_copies*len(self.T.MTRAIN))]
    self.MValAnswers = [[None, None]
        for _ in range(self.mtrain_copies*len(self.T.MTRAIN))]
    self.MTestAnswers = [[None, None]
        for _ in range(self.mtrain_copies*len(self.T.MTRAIN))]
    self.OldMTrainAnswers = [[None, None]
        for _ in range(self.mtrain_copies*len(self.T.MTRAIN))]

  #######################################
  ###                                 ###
  ### BOUNCEGRAD: SAnnealing with SGD ###
  ###                                 ###
  #######################################

  def run_MAML(self, structure, dataset):
    '''
    Run MAML of structure 'structure' on 'dataset'.

    Code inspired by a single iteration of:
    https://github.com/katerakelly/pytorch-maml/blob/master/src/maml.py#L149
    '''
    self.slow_net = self.S.composer_class(composer=self.composer,
      module_list=self.L, loss_fn=None, structure=structure)
    baseComposer=self.S.composer_class(composer=self.composer,
      module_list=self.L, loss_fn=None, structure=structure)
    self.fast_net = InnerLoop(baseComposer=baseComposer, module_list=self.L,
        loss_fn=self.MAML_loss_fn, num_updates=self.MAML_inner_updates,
        step_size=self.MAML_step_size)
    self.slow_net.cuda()
    self.fast_net.copy_weights(self.slow_net)
    metrics, g = self.fast_net.forward(dataset)
    (train_loss, val_loss, train_ans, val_ans) = metrics
    return train_loss, val_loss, train_ans, val_ans, g

  def run_model(self, structure, inp, instructions={}):
    '''
    Returns the composition of several modules
    '''
    if len(inp.shape) == 0 or inp.shape[0]==0: return inp #empty tensor
    if len(inp.shape) == 1: inp = inp.unsqueeze(1)
    return self.S.composer_class(composer=self.composer,
        module_list=self.L, structure=structure,instructions=instructions)(inp)

  def evaluate(self, structure, dataset, mode):
    '''
    Evaluates the dataset according to the structure
    '''
    inp = dataset.TrainInput if mode=='Train' else dataset.ValInput
    out = (dataset.TrainOutput if mode=='Train' else dataset.ValOutput)
    if len(inp.shape) == 0 or inp.shape[0] == 0: #empty tensors
      return torch.FloatTensor(np.array([0])), out
    pred = self.run_model(structure,inp)
    return F.mse_loss(pred, out), pred

  def evaluate_several_structures(self, num, keep_last=False, mode='Train'):
    '''
    Returns the mean loss for several random datasets
    keep_last: evaluate same datasets as last time
    '''
    if not keep_last:
      self.several_ds_to_eval = np.random.choice(num, size=len(self.T.MTRAIN))
    res = 0.
    for i in self.several_ds_to_eval:
      original_train, original_train_ans = self.evaluate(
          self.S.TrainStructures[i], self.T.MTRAIN[i], mode)
      res += original_train.data.cpu().numpy()
    return res/len(self.several_ds_to_eval)

  def bounce(self, structure, dataset, temp, do_grad):
    '''
    Propose a modification to structure
    Do a SA step depending on performance in Train
    Backpropagate on Val
    '''
    if self.do_bounce:
      new_structure = copy.deepcopy(structure)
      self.S.propose_new_structure(new_structure)
    ##################################
    # Simulated Annealing comparison #
    ##################################
    if self.MAML:
      (original_train, original_val,
          original_train_ans, original_val_ans, original_gradient) = (
          self.run_MAML(structure, dataset))
      original_train_np = original_train.data.cpu().numpy()
      if self.do_bounce:
        new_train, new_val, new_train_ans, new_val_ans, new_gradient  = (
            self.run_MAML(new_structure, dataset))
        new_train_np = new_train.data.cpu().numpy()
    else:
      MAML_g = None
      original_train, original_train_ans = self.evaluate(structure,
          dataset, 'Train')
      original_train_np = original_train.data.cpu().numpy()
      if self.do_bounce:
        new_train, new_train_ans = self.evaluate(new_structure, dataset,
            'Train')
        new_train_np = new_train.data.cpu().numpy()

    if self.do_bounce:
      upt_factor = min(0.01, self.SA_running_acc_rate/self.SA_running_factor)
      prob_accept = np.exp((original_train_np - new_train_np)/temp)
    if (self.do_bounce and ( #Accept
      new_train_np <= original_train_np or np.random.rand() < prob_accept)):
      if original_train_np < new_train_np: #update running frac of worse accepts
        self.SA_running_factor = ((1-upt_factor)*self.SA_running_factor +
            upt_factor)
        self.SA_running_acc_rate = ((1-upt_factor)*self.SA_running_acc_rate +
            upt_factor)
      if self.MAML: MAML_g = new_gradient
      else:
        if self.LocalOpt is not None: new_train.backward()
        # structure['node_positions'].requires_grad = False
        self.S.update_structure(structure)
        new_val, new_val_ans = self.evaluate(new_structure, dataset, 'Val')
        if do_grad: new_val.backward()
        # structure['node_positions'].requires_grad = True
      return (new_structure, new_train, new_val,
          new_train_ans.data.cpu().numpy(),
          new_val_ans.data.cpu().numpy(), MAML_g)
    else: #Reject
      if self.do_bounce and original_train_np < new_train_np:
        #update running frac of worse accepts
        self.SA_running_factor = ((1-upt_factor)*self.SA_running_factor +
            upt_factor)
        self.SA_running_acc_rate = (1-upt_factor)*self.SA_running_acc_rate
      if self.MAML: MAML_g = original_gradient
      else:
        if self.LocalOpt is not None: original_train.backward()
        # structure['node_positions'].requires_grad = False
        self.S.update_structure(structure)
        original_val, original_val_ans = self.evaluate(structure, dataset,
            'Val')
        if do_grad: original_val.backward()
        # structure['node_positions'].requires_grad = True
      return (structure, original_train, original_val,
          original_train_ans.data.cpu().numpy(),
          original_val_ans.data.cpu().numpy(), MAML_g)

  ##############################
  ## MAIN BOUNCEGRAD FUNCTION ##
  ##############################
  def SAConfig_SGDModules(self, optimization_steps):
    '''
    Optimization by Simulated Annealing on the module configurations
    and SGD (Adam in our case) on the modules at every step.
    '''
    # Cooling schedule decreases exponentially wrt the fraction of accepted
    # proposals with worse performance, starting at 1 (accept no matter what)
    # Therefore we have to keep a running estimate of fraction of accepts.
    self.SA_running_acc_rate = 1e-9 #initial counters for Simulated Annealing
    self.SA_running_factor = 1e-9 #normalizing constant
    temp = np.exp(self.initial_temp) #temperature in the SA formula
    temp_change = 1.1
    CosDist = nn.CosineSimilarity(dim=1)
    #############################
    # Create initial structures #
    #############################
    self.S.initialize_all_structures(T=self.T, mtrain_copies=self.mtrain_copies)
    if len(self.S.StructureParameters):
      self.LocalOpt = torch.optim.Adam(self.S.StructureParameters,
          lr=self.meta_lr/10)
      self.LocalOpt_scheduler = ReduceLROnPlateau(
        optimizer=self.SOpt, factor=1/2.,
        mode='min', patience=20, threshold=0, cooldown=100,
        verbose=True, min_lr=1e-4)
    if self.load_structures_and_metrics != '':
      self.load_strmet(self.load_structures_and_metrics)

    # input('Check structure: '+ str( self.S.TrainStructures[0]))
    for step in Tqdm(range(optimization_steps)):
      self.step = step * self.mtrain_copies
      if step:
        self.SOpt_scheduler.step(self.mean_current_val) #MTrain-Val
        if self.LocalOpt is not None:
          self.LocalOpt_scheduler.step(self.mean_current_val) #MTrain-Val
      self.METRICS['TrainStructures'] = self.S.TrainStructures
      self.METRICS['ValStructures'] = self.S.ValStructures
      self.S.update_PosUsage_counters(METRICS=self.METRICS)
      self.S.update_Usage_counters(METRICS=self.METRICS, T=self.T)
      self.S.update_customized_counters(METRICS=self.METRICS)
      self.update_Sharing_counters()

      #with default values the midpoint @0.7%, end @0.005%
      acc_rate = np.exp(self.initial_acc-5.*step/optimization_steps)
      if self.SA_running_acc_rate/self.SA_running_factor < acc_rate:
        temp *= temp_change
      else: temp /= temp_change

      self.current_train = []
      self.current_val = []
      self.MTrain_norm_diff = []
      self.MTrain_cos_diff = []
      if self.MAML: maml_gradients = []
      for i, structure in Tqdm(enumerate(self.S.TrainStructures)):
        dataset = self.T.MTRAIN[i%len(self.T.MTRAIN)] #multiple structures per dataset
        #Update structure
        self.S.update_structure(self.S.TrainStructures[i], step=self.step)
        #######################
        # Simulated Annealing #
        #######################
        (self.S.TrainStructures[i], train_loss, val_loss, train_ans, val_ans,
            MAML_g) = self.bounce(structure, dataset, temp, do_grad=True)
        self.current_train.append(train_loss.data.cpu().numpy())
        self.current_val.append(val_loss.data.cpu().numpy())
        if self.MTrainAnswers[i][0] is None:
          self.MTrainAnswers[i][0] = train_ans * self.ans_eps
        else: self.MTrainAnswers[i][0] = (
            (1-self.ans_eps)*self.MTrainAnswers[i][0] +
            train_ans * self.ans_eps)
        if self.MTrainAnswers[i][1] is None:
          self.MTrainAnswers[i][1] = val_ans * self.ans_eps
        else:
          self.MTrainAnswers[i][1] = ((1-self.ans_eps)
              *self.MTrainAnswers[i][1] + val_ans*self.ans_eps)
        if self.OldMTrainAnswers[i][0] is not None:
          self.MTrain_norm_diff.append(
              np.mean(np.linalg.norm(
                self.OldMTrainAnswers[i][0]-train_ans,axis=1)))
          self.MTrain_cos_diff.append(
              torch.mean(CosDist(torch.FloatTensor(
                self.OldMTrainAnswers[i][0]-
                dataset.TrainOutput.data.cpu().numpy()),
                torch.FloatTensor(
                  train_ans-dataset.TrainOutput.data.cpu().numpy()))))
        self.OldMTrainAnswers[i][0] = train_ans

        ####################
        # Gradient Descent #
        ####################
        if self.MAML: maml_gradients.append(MAML_g)
        if (i == len(self.S.TrainStructures)-1 or (self.execute_gd_every>0 and
          i % self.execute_gd_every == self.execute_gd_every -1)):
          if self.MAML:
            # Compute sum of maml_gradients to each module
            # Inspired by:
            # github.com/katerakelly/pytorch-maml/blob/master/src/maml.py#L66
            # but multiple changes because they have a single structure
            self.dict_gradients = {}
            for G in maml_gradients:
              for (key, value) in G.items():
                if value is None: continue
                name = '.'.join(key.split('.')[1:])
                if name not in self.dict_gradients:
                  self.dict_gradients[name] = value
                else: self.dict_gradients[name] += value

            hooks = []
            for (k,v) in self.L.named_parameters():
              def get_closure():
                key = k
                value = v
                def replace_grad(grad):
                  if key in self.dict_gradients:
                    return self.dict_gradients[key]
                  else: return torch.zeros_like(value)
                return replace_grad
              hooks.append(v.register_hook(get_closure()))
            self.SOpt.zero_grad()
            for module in self.L:
              dummy_loss = module.dummy_forward_pass()
              dummy_loss.backward()
            self.SOpt.step()
            for h in hooks: h.remove()

            maml_gradients = []
          else: #No MAML --> use regular loss
            self.SOpt.step()
            self.SOpt.zero_grad()
            if self.LocalOpt is not None:
              self.LocalOpt.step()
              self.LocalOpt.zero_grad()

      #Simulated Annealing on MetaValidation data
      self.current_Mtrain = []
      self.current_Meval = []
      for i, structure in enumerate(self.S.ValStructures):
        dataset = self.T.MVAL[i%len(self.T.MVAL)]
        #Update structure
        self.S.update_structure(self.S.ValStructures[i], step=self.step)
        (self.S.ValStructures[i], train_loss, val_loss,
            train_ans, val_ans, MAML_g) = (
            self.bounce(structure, dataset, temp, do_grad=False))
        self.current_Mtrain.append(train_loss.data.cpu().numpy())
        self.current_Meval.append(val_loss.data.cpu().numpy())
        if self.MValAnswers[i][0] is None:
          self.MValAnswers[i][0] = train_ans * self.ans_eps
        else:
          self.MValAnswers[i][0] = (
              (1-self.ans_eps)*self.MValAnswers[i][0] + train_ans*self.ans_eps)
        if self.MValAnswers[i][1] is None:
          self.MValAnswers[i][1] = val_ans * self.ans_eps
        else:
          self.MValAnswers[i][1] = ((1-self.ans_eps)*self.MValAnswers[i][1] +
              val_ans * self.ans_eps)
      #Zero-out optimizers (step in MTRAIN performed + dont want step from MVAL)
      self.SOpt.zero_grad()
      ###################
      # Stats and plots #
      ###################
      self.answers_running_score = (self.answers_running_score*
          (1.-self.ans_eps) + self.ans_eps)
      ensemble_train = np.mean([np.mean(
          (self.MTrainAnswers[i][0]/self.answers_running_score
            -self.T.MTRAIN[i].TrainOutput.data.cpu().numpy())**2)
          for i in range(self.T.mtrain)])
      ensemble_val = np.mean([np.mean(
        (self.MTrainAnswers[i][1]/self.answers_running_score
          -self.T.MTRAIN[i].ValOutput.data.cpu().numpy())**2)
        for i in range(self.T.mtrain)])
      ensemble_Mtrain = np.mean([np.mean(
        (self.MValAnswers[i][0]/self.answers_running_score
          -self.T.MVAL[i].TrainOutput.data.cpu().numpy())**2)
        for i in range(self.T.mval)])
      ensemble_Mval = np.mean([np.mean(
        (self.MValAnswers[i][1]/self.answers_running_score
          -self.T.MVAL[i].ValOutput.data.cpu().numpy())**2)
        for i in range(self.T.mval)])
      self.writer.add_scalar('ensemble/train', ensemble_train.item(),
          self.step)
      self.writer.add_scalar('ensemble/val', ensemble_val.item(),
          self.step)
      self.writer.add_scalar('ensemble/Mtrain', ensemble_Mtrain.item(),
          self.step)
      self.writer.add_scalar('ensemble/Mval', ensemble_Mval.item(),
          self.step)
      self.writer.add_scalar('SA/temp', np.log10(temp).item(),
          self.step)
      self.writer.add_scalar('SA/acc_rate',
          np.log10(self.SA_running_acc_rate).item(), self.step)
      self.update_stats()
      self.S.update_customized_stats(self)
      if  step % self.plot_freq == 0 or step==self.optimization_steps-1:
        #plot & store metrics to JSON
        self.log_metrics()
        self.store_simple_metrics()
        # self.store_metrics()
        if self.save_modules != '': self.save_L(self.save_modules)

  #########################################
  # Logistic functions of little interest #
  #########################################
  def save_L(self, filepath=None):
    '''
    Saves ModuleList
    '''
    if filepath is None: filepath = 'moduleList-'
    for i_m, module in enumerate(self.L):
      torch.save(module.state_dict(), filepath+str(i_m))

  def load_L(self, filepath=None):
    '''
    Loads ModuleList
    '''
    if filepath is None: filepath = 'moduleList-'
    for i_m, module in enumerate(self.L):
      self.L[i_m].load_state_dict(torch.load(filepath+str(i_m)))

  def load_strmet(self, filepath=None):
    '''
    Loads structures and metrics
    '''
    if filepath is None: filepath = 'metrics/'
    if os.path.isdir(filepath):
      filepath = os.path.join(filepath, 'metrics.json')
    with open(filepath, 'r') as infile:
      self.METRICS = json.load(infile)
    self.S.TrainStructures = self.METRICS['TrainStructures']
    self.S.ValStructures = self.METRICS['ValStructures']

  def update_stats(self):
    '''
    Updates several metrics after each step
    '''
    #Differences between timesteps
    if self.MTrain_norm_diff != []:
      self.writer.add_scalar('norm_diff/perc_10',
          np.percentile(self.MTrain_norm_diff, 10).item(), self.step)
      self.writer.add_scalar('norm_diff/perc_90',
          np.percentile(self.MTrain_norm_diff, 90).item(), self.step)
      self.writer.add_scalar('norm_diff/perc_10',
          np.percentile(self.MTrain_cos_diff, 10).item(), self.step)
      self.writer.add_scalar('norm_diff/perc_90',
          np.percentile(self.MTrain_cos_diff, 90).item(), self.step)

    #Weight norms
    self.act_norms = [torch.norm(_) for m in self.L for _ in m.parameters()]
    self.norm_ratios = [(a/b).detach().cpu().numpy() for (a,b) in zip(
      self.act_norms, self.initial_norms)]
    self.writer.add_scalar('norm_ratios/mean',
        np.mean(self.norm_ratios).item(), self.step)
    self.writer.add_scalar('norm_ratios/std',
        np.std(self.norm_ratios).item(), self.step)
    #Error metrics
    self.mean_current_val = np.mean(self.current_val).item()
    self.writer.add_scalar('mean_loss/T_val', self.mean_current_val,
        self.step)
    self.writer.add_scalar('std_loss/T_val',
        np.std(self.current_val).item(), self.step)
    self.writer.add_scalar('min_loss/T_val',
        np.min(self.current_val).item(), self.step)
    if len(self.current_train):
      self.writer.add_scalar('mean_loss/T_train',
          np.mean(self.current_train).item(), self.step)
      self.writer.add_scalar('std_loss/T_train',
          np.std(self.current_train).item(), self.step)
      self.writer.add_scalar('min_loss/T_train',
          np.min(self.current_train).item(), self.step)
    if len(self.current_Mtrain):
      self.writer.add_scalar('mean_loss/V_train',
          np.mean(self.current_Mtrain).item(), self.step)
      self.writer.add_scalar('std_loss/V_train',
          np.std(self.current_Mtrain).item(), self.step)
      self.writer.add_scalar('min_loss/V_train',
          np.min(self.current_Mtrain).item(), self.step)
    if len(self.current_Meval):
      self.writer.add_scalar('mean_loss/V_val',
          np.mean(self.current_Meval).item(), self.step)
      self.writer.add_scalar('std_loss/V_val',
          np.std(self.current_Meval).item(), self.step)
      self.writer.add_scalar('min_loss/V_val',
          np.min(self.current_Meval).item(), self.step)

  def is_jsonable(self, x):
    try:
      json.dumps(x)
      return True
    except: return False

  def store_simple_metrics(self):
    '''
    Stores all metrics that can be JSON-ed to a JSON file
    '''
    self.METRICS['params'] = {attr:getattr(self, attr) for attr in
        dir(self) if self.is_jsonable(getattr(self, attr))}
    with open(os.path.join(self.plot_name,
      'metrics.json'), 'w') as outfile:
      json.dump({k: v for k,v in self.METRICS.items()
      if self.is_jsonable(v)}, outfile)

  def store_metrics(self):
    '''
    Stores all the metrics to a JSON file
    '''
    self.METRICS['time'] = int(time.time())
    self.METRICS['params'] = {attr:getattr(self, attr) for attr in dir(self)
        if (type(getattr(self, attr)) in
          [type(1), type(1.0), type('a'), type(None)]
          or (type(getattr(self, attr))==type([]) and
            len(getattr(self, attr)) and type(getattr(self, attr)[0]) in
            [type(1), type(1.0), type('a'), type(None)])
          or (type(getattr(self, attr))==type({}) and
            len(getattr(self, attr)) and
            type(getattr(self, attr)[list(getattr(self,attr).keys())[0]]) in
            [type(1), type(1.0), type('a'), type(None)])
        and not callable(getattr(self, attr)) and not attr.startswith('__'))}
    with open('metrics/'+self.plot_name[:-1]+'.json', 'w') as outfile:
      json.dump(self.METRICS, outfile)
    with open(self.plot_name+'metrics.json', 'w') as outfile:
      json.dump(self.METRICS, outfile)

  def plot_sharing(self):
    if (self.METRICS['Sharing'] is not None and
            np.max(self.METRICS['Sharing'])>1e-4 and
            len(self.METRICS['NumberToWords']) <= 50):
      #Find ordering
      aux = list(enumerate(self.METRICS['NumberToWords']))
      aux.sort(key = lambda x : x[1])
      sorted_order = [_[0] for _ in aux]
      cax = plt.gca().matshow(np.array(
        self.METRICS['Sharing'])[sorted_order,:][:,sorted_order]
        /self.S.usage_normalization)
      plt.gca().set_xticklabels(['']+sorted(self.METRICS['NumberToWords']))
      plt.gca().set_yticklabels(['']+sorted(self.METRICS['NumberToWords']))
      plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
      plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
      if self.store_video:
        plt.savefig(os.path.join(self.plot_name, 'video/sharing-rate_'+
          str(self.step)))
      plt.gcf().colorbar(cax)
      plt.savefig(os.path.join(self.plot_name, 'sharing-rate'))
      plt.clf()

  def log_metrics(self):
    '''
    Creates all plots based on self.METRICS.
    '''
    # Plot Usage rate
    self.S.plot_usage(directory=self.plot_name)
    self.S.plot_customized_usage_rate(directory=self.plot_name)
    self.plot_sharing()

    # Plot basic modules
    if self.T.MTRAIN[0].TrainInput.shape[-1] > 1: return
    print_plot = 0
    input_range = np.linspace(-1,1).reshape((-1,1))
    norm_input_range = self.T.normalize_input(input_range)
    norm_input_torch = torch.FloatTensor(norm_input_range).to(
        device=self.nn_device)
    if self.perm_sample_modules is None:
      self.perm_sample_modules=np.random.choice(len(self.L),
          min(len(self.L), 9), replace=False)
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(min(len(self.L), 9)):
      net = self.L[self.perm_sample_modules[i]]
      color = 'b'
      if net.inp == 1 and net.out == 1: #plotable function
        ax[i//3, i%3].plot(input_range,
            2.*self.T.denormalize_output(
              net(norm_input_torch).data.cpu().numpy()), c=color)
        ax[i//3, i%3].set_ylim([-1.5,1.5])
        ax[i//3, i%3].set_xlim([-1,1])
        ax[i//3, i%3].set_xticks(np.array([-1,0,1]))
        ax[i//3, i%3].set_yticks(np.array([-1,0,1]))
        print_plot += 1
    if print_plot > 0:
      plt.savefig(os.path.join(self.plot_name, 'sample-modules'))
      if self.store_video:
        plt.savefig(os.path.join(self.plot_name,
          'video/modules_'+str(self.step)))
      plt.cla()
    # Plot basic comparisons
    fig, ax = plt.subplots(nrows=3, ncols=3)
    if self.perm_sample_fns is None:
      self.perm_sample_fns=np.random.choice(len(self.T.MTRAIN),
          min(len(self.T.MTRAIN), 9), replace=False)
    for i in range(min(len(self.T.MTRAIN), 9)):
      dataset = self.T.MTRAIN[self.perm_sample_fns[i]]
      ax[i//3, i%3].scatter(dataset.UValInput,
          dataset.UValOutput, c='g', label='val')
      ax[i//3, i%3].scatter(dataset.UTrainInput,
          dataset.UTrainOutput, c='r', label='train',
          s=18)
      if self.MAML:
        ax[i//3, i%3].scatter(self.T.MTRAIN[self.perm_sample_fns[i]].UValInput,
            self.T.denormalize_output(
              self.MTrainAnswers[self.perm_sample_fns[i]][1]))
      else:
        structure = self.S.TrainStructures[self.perm_sample_fns[i]]
        structure_output = self.T.denormalize_output(self.run_model(structure,
            norm_input_torch).data.cpu().numpy())
        ax[i//3, i%3].plot(input_range, structure_output)
      ax[i//3, i%3].set_xlim([
        np.floor(np.min(self.T.MTRAIN[self.perm_sample_fns[i]].UValInput)),
        np.ceil( np.max(self.T.MTRAIN[self.perm_sample_fns[i]].UValInput))])
      ax[i//3, i%3].set_ylim([
        np.floor(np.min(self.T.MTRAIN[self.perm_sample_fns[i]].UValOutput))-.5,
        np.ceil( np.max(self.T.MTRAIN[self.perm_sample_fns[i]].UValOutput))+.5])
      ax[i//3, i%3].set_xticks(np.array([-1,0,1]))
      ax[i//3, i%3].set_yticks(np.array([-1,0,1]))
    plt.savefig(os.path.join(self.plot_name, 'comparisons'))
    if self.store_video:
      plt.savefig(os.path.join(self.plot_name,
        'video/comparisons_'+str(self.step)))
    plt.clf()

  def update_Sharing_counters(self):
    '''
    Updates table of E[# of modules shared by 2 keywords]
    More precisely dataset names are a list of keywords A_B_C
    the entries of the table are those keywords.

    For example:
     - square_plywood in the MIT dataset --> object=square,surface=plywood
     - 1_4 in the Berkeley dataset --> action 1 actor 5
    These plots show what structure the modules capture
    '''
    if self.meta_lr == 0 or self.mtrain_copies > 1:
      if self.step == 0: print('Not doing Sharing for now')
      return
    if len(self.METRICS['WordsToNumber']) > 50: return #too many keywords
    eps = 1e-3
    self.S.usage_normalization = self.S.usage_normalization*(1-eps) + eps
    for i_s, i_structure in enumerate(
        self.S.TrainStructures+self.S.ValStructures):
      for j_s, j_structure in enumerate(
          self.S.TrainStructures+self.S.ValStructures):
        #count number of matches
        i_modules = self.S.modules_given_structure(i_structure)
        j_modules = self.S.modules_given_structure(j_structure)
        count = 0
        for a in set(i_modules):
          count += min(i_modules.count(a), j_modules.count(a))
        count /= len(i_modules)
        if i_s < self.T.mtrain: i_words = self.T.MTRAIN[i_s].name.split('_')
        else: i_words = self.T.MVAL[i_s-self.T.mtrain].name.split('_')
        if j_s < self.T.mtrain: j_words = self.T.MTRAIN[j_s].name.split('_')
        else: j_words = self.T.MVAL[j_s-self.T.mtrain].name.split('_')
        for i_w in i_words + j_words:
          if i_w not in self.METRICS['WordsToNumber']:
            self.METRICS['WordsToNumber'][i_w] = (
                len(self.METRICS['WordsToNumber']))
            self.METRICS['NumberToWords'].append(i_w)
            if len(self.METRICS['WordsToNumber']) > 50:return#too many keywords
        for i_w in i_words:
          for j_w in j_words:
            (ni_w, nj_w) = (self.METRICS['WordsToNumber'][i_w],
                self.METRICS['WordsToNumber'][j_w])
            self.METRICS['Sharing'][ni_w][nj_w] *= (1.-eps)
            self.METRICS['Sharing'][nj_w][ni_w] *= (1.-eps)
            self.METRICS['Sharing'][ni_w][nj_w] += count*eps
            self.METRICS['Sharing'][nj_w][ni_w] += count*eps
