import argparse
from modular_metalearning import BounceGrad
from hgnn_composer import HGNN_Structure

def main():
  #########
  # Flags #
  #########
  parser = argparse.ArgumentParser()
  # Compute flags
  parser.add_argument('--torch_seed', dest='torch_seed', default=0,
      help='pytorch seed')
  parser.add_argument('--device', dest='nn_device', default='cuda:0',
      help='what device we want to run things in')
  parser.add_argument('--cpu_threads', dest='cpu_threads', default=-1,
          help='number of cpu threads [default: do not set]')
  parser.add_argument('--stop_edge_acc', dest='stop_edge_acc', type=int,
          default=-1,
          help='minutes until <1% increase in edge accuracy results in kill')
  # Data flags
  parser.add_argument('--data_desc', dest='data_desc',
      help='description of data; e.g. "charged5"')
  parser.add_argument('--limit_data', dest='limit_data', type=int,
      help='maximum number of points per dataset', default=10000)
  parser.add_argument('--repeat_mtrain', dest='repeat_mtrain', type=int,
      default=0, help='number of times to repeat meta training data')
  parser.add_argument('--max_datasets', dest='max_datasets', type=int,
      default=100000, help='maximum number of datasets')
  parser.add_argument('--data_split', dest='data_split', default='20,80,0',
      help='comma-separated distribution (in %) of train,val,test per dataset')
  parser.add_argument('--meta_split', dest='meta_split', default='90,10,0',
      help='comma-separated distribution (in %) of mtrain,mval,mtest')
  parser.add_argument('--split_by_file', dest='split_by_file',
      action='store_true')
  parser.add_argument('--smaller_MVals', dest='smaller_MVals', type=str,
      default='', help='List of extra smaller training sizes for MVal')
  parser.add_argument('--dont_normalize', dest='normalize_data',
      action='store_false')
  parser.add_argument('--normalize_output', dest='normalize_output',
      action='store_true')
  parser.add_argument('--input_noise', dest='input_noise', type=float,
          default=0., help='amount of noise to add to MTRAIN input')

  # BounceGrad flags
  parser.add_argument('--dont_bounce', dest='do_bounce', action='store_false',
      help='Skip Simulated Annealing and proposing structures; only Grad step')
  parser.add_argument('--dont_accept', dest='dont_accept', action='store_true',
      help='Never accept new structures; useful for debugging and plotting')
  parser.add_argument('--meta_lr', dest='meta_lr', type=float, default='1e-3',
      help='learning rate for module parameters')
  parser.add_argument('--encoder_lr', dest='encoder_lr', type=float,
          default='1e-3', help='learning rate for encoder')
  parser.add_argument('--num_modules', dest='num_modules',
      help='comma-separated list with size of each population of module type')
  parser.add_argument('--type_modules', dest='type_modules',
      help='comma-separated list describing the type of each module category')
  parser.add_argument('--composer', dest = 'composer', default='composition',
      help='Which type of composition to use; \
          for example "compositon,sum,concatenate,gnn"')
  parser.add_argument('--optimization_steps', dest='optimization_steps',
      type=int, default = 1000, help='number of BounceGrad steps')
  parser.add_argument('--temp_slope_opt_steps', dest='temp_slope_opt_steps',
      type=int, default = -1, help='number of BounceGrad steps')
  parser.add_argument('--meta_batch_size', dest='meta_batch_size', type=int,
      default=0, help='Number of metatrain cases between gradient steps;\
          0 if all MTRAIN')
  parser.add_argument('--auto_temp', dest='auto_temp',
          action='store_true',
          help='anytime temperature schedule based on current train loss')

  # Plotting flags
  parser.add_argument('--plot_name', dest='plot_name',
      default='default', help='Name for error plot')
  parser.add_argument('--store_video', dest='store_video', action='store_true',
       help='Store images every step into video folder')
  parser.add_argument('--plot_ymax', dest='plot_ymax', type=float,
      default = -1., help='maximum y in zoomed loss plot')
  parser.add_argument('--plot_freq', dest='plot_freq', type=int,
      default=5, help='Number of optimization steps between plots')
  parser.add_argument('--plot_with_without_node',
          dest='plot_with_without_node', action='store_true',
          help='make plots with predictions and predictions blanking\
          out 1st node')


  # Flags mostly useful for restarting experiments
  parser.add_argument('--load_modules', dest='load_modules', type=str,
      default='', help='Filepath to load modules from self.L;\
          empty if dont want to load')
  parser.add_argument('--load_structures_and_metrics',
      dest='load_structures_and_metrics', type=str, default='',
      help='Filepath to load structures and metrics;\
          empty if dont want to load')
  parser.add_argument('--save_modules', dest='save_modules', type=str,
      default='', help='Filepath to save modules from self.L;\
          if empty use plot_name')
  parser.add_argument('--initial_temp', dest='initial_temp', type=float,
      default = 0, help='[log] initial temperature')
  parser.add_argument('--min_temp', dest='min_temp', type=float,
      default = -100., help='minimum [log] temperature')
  parser.add_argument('--initial_acc', dest='initial_acc', type=float,
      default = 0, help='[log] initial acceptance ratio')
  parser.add_argument('--skip_mtrain', dest='do_mtrain', action='store_false',
          help='Skip meta training for evaluating meta test only')

  parser.add_argument('--find_node', dest='find_node', action='store_true',
          help='mode to optimize trajectory of 1st node')

  # metrics/models path
  parser.add_argument('--log_dir', dest='log_dir', default='runs', type=str, help='where to save metrics')
  parser.add_argument('--log_dir_comment', dest='log_dir_comment', type=str, default='',
                      help='comment to append to metrics file')
  parser.add_argument('--models_path', dest='models_path', default='models', type=str, help='where to save model checkpoints')

  # GNN proposal function
  parser.add_argument('--encoder', dest='encoder_input_type',
          default='state', type=str, help='encoder input type to use')
  parser.add_argument('--encoder_type', dest='encoder_type', default='cnn', type=str, help='encoder type to use')
  parser.add_argument('--encoder_hs', dest='encoder_hs', default=512, type=int, help='size of encoder hidden layer')
  parser.add_argument('--train_only_acc', dest='enc_train_only_acc', action='store_true',
                      help='train encoder with only accepts')
  parser.add_argument('--structure_enc_input', dest='structure_enc_input', default='uniform',
                      help='which input method to use for structure encoder, choose from "uniform" and "flip"')
  parser.add_argument('--proposal_type', dest='proposal_type', default='edge',
                      help='during new structure proposal, whether to draw a single edge to change'
                           'or to draw all edges for a single node to change. choose from "node" and "edge"')
  parser.add_argument('--proposal_flip', dest='proposal_flip', action='store_true',
                      help='for proposal type "node", whether to flip all edges or randomly sample.')


  # Parsing args
  args = parser.parse_args()

  # Finding composer
  composer = args.composer
  if composer.startswith('gnn'): S = GNN_Structure(args=args)
  elif composer.startswith('hgnn'): S = HGNN_Structure(args=args)
  # elif composer == 'distributedDL': S = DistributedDL_Structure(args=args)
  elif composer.startswith('sum'):
    [composer, args.structure_size] = composer.split('-')
    args.structure_size=int(args.structure_size)
    S = Sum_Structure(args=args)
  elif composer.startswith('functionComposition'):
    [composer, args.structure_size] = composer.split('-')
    args.structure_size=int(args.structure_size)
    S = FunctionComposition_Structure(args=args)
  else: raise NotImplementedError

  bg = BounceGrad(S=S, args=args)
  bg.SAConfig_SGDModules(args.optimization_steps)

if __name__ == '__main__':
  main()
