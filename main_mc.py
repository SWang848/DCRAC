import sys
import numpy as np
from optparse import OptionParser

from minecart import Minecart
from agent import *
from utils import *

AGENT_DICT = {
    'DCRAC': DCRACAgent,
    'DCRACS': DCRACSAgent,
    'DCRAC0': DCRAC0Agent, 
    'CNLSTM': CN0Agent, 
    'CN': CNAgent,
}

mkdir_p('output')
mkdir_p('output/logs')
mkdir_p('output/networks')
mkdir_p('output/pred')
mkdir_p('output/imgs')

parser = OptionParser()
parser.add_option('-a', '--agent', dest='agent', choices=['DCRAC', 'DCRACS', 'DCRAC0', 'CNLSTM', 'CN'], default='DCRACS')
parser.add_option('-n', '--net-type', dest='net_type', choices=['R', 'M', 'F'], default='R', help='Agent architecture type: [R]ecurrent, [M]emNN or [F]C')
parser.add_option('-r', '--replay', dest='replay', default='DER', choices=['STD', 'DER'], help='Replay type, one of "STD","DER"')
parser.add_option('-s', '--buffer-size', dest='buffer_size', default='100000', help='Replay buffer size', type=int)
parser.add_option('-m', '--memnn-size', dest='memnn_size', default='10', help='Memory network memory size', type=int)
parser.add_option('-d', '--dupe', dest='dupe', default='CN', choices=['CN', 'NONE'], help='Extra training')
parser.add_option('-t', '--timesteps', dest='timesteps', default='10', help='Recurrent timesteps', type=int)
parser.add_option('-e', '--end_e', dest='end_e', default='0.05', help='Final epsilon value', type=float)
parser.add_option('-l', '--lr-c', dest='lr_c', default='0.02', help='Critic learning rate', type=float)
parser.add_option('-L', '--lr-a', dest='lr_a', default='0.001', help='Actor learning rate', type=float)
parser.add_option('--buffer-a', dest='buffer_a', default='2.', help='Reply buffer error exponent', type=float)
parser.add_option('--buffer-e', dest='buffer_e', default='0.01', help='Reply buffer error offset', type=float)
parser.add_option('-u', '--update-period', dest='updates', default='4', help='Update interval', type=int)
parser.add_option('-f', '--frame-skip', dest='frame_skip', default='4', help='Frame skip', type=int)
parser.add_option('-b', '--batch-size', dest='batch_size', default='32', help='Sample batch size', type=int)
parser.add_option('-g', '--discount', dest='discount', default='0.98', help='Discount factor', type=float)
parser.add_option('--anneal-steps', dest='steps', default='100000', help='Steps',  type=int)
parser.add_option('-c', '--mode', dest='mode', choices=['regular', 'sparse'], default='sparse')
parser.add_option('-v', '--obj-func', dest='obj_func', choices=['td', 'q', 'y'], default='td')
parser.add_option('--gpu', dest='gpu_setting', choices=['1', '2', '3'], default='3', help='1 for CPU, 2 for GPU, 3 for CuDNN')
parser.add_option('--log-game', action='store_true', dest='log_game')


(options, args) = parser.parse_args()

hyper_info = '{}-r{}{}-d{}-t{}-batsiz{}-lr{}-{}-eval{}'.format(
    options.alg, options.replay, str(options.buffer_size), options.dupe, options.timesteps, 
    options.batch_size, str(options.lr_c), options.mode, options.obj_func)

# create evironment
json_file = 'mine_config.json'
env = Minecart.from_json(json_file)

all_weights = get_weights_from_json('./train_weights_minecart.json')

log_file = open('output/logs/mc_rewards_{}.log'.format(hyper_info), 'w', 1)

deep_agent = AGENT_DICT[options.agent]
agent = deep_agent(env, 
                   gamma=options.discount,
                   weights=None,
                   timesteps=10,
                   batch_size=options.batch_size,
                   replay_type=options.replay,
                   buffer_size=options.buffer_size,
                   buffer_a=options.buffer_a,
                   buffer_e=options.buffer_e,
                   memnn_size=options.memnn_size,
                   end_e=options.end_e,
                   net_type=options.net_type,
                   obj_func=options.obj_func,
                   lr=options.lr_c,
                   lr_2=options.lr_a,
                   frame_skip=options.frame_skip,
                   update_interval=options.updates,
                   dupe=None if options.dupe == 'NONE' else options.dupe,
                   extra=hyper_info,
                   gpu_setting=options.gpu_setting)

steps_per_weight = 50000 if options.mode == 'sparse' else 1
agent.train(log_file, options.steps, all_weights, steps_per_weight, options.steps*10, log_game_step=options.log_game)