import sys
import time
import numpy as np
from optparse import OptionParser

from deep_sea_treasure import DeepSeaTreasure
from agentT import DCRACSAgent, DCRACAgent, DCRACSEAgent, DCRAC0Agent, CNAgent, CN0Agent
# from agentT import DCRACSRunAgent, CNRunAgent
from utils import mkdir_p, get_weights_from_json
from stats import rebuild_log, print_stats, compute_log

AGENT_DICT = {
    'DCRAC': DCRACAgent,
    'DCRACS': DCRACSAgent,
    'DCRACSE': DCRACSEAgent,
    'DCRAC0': DCRAC0Agent, 
    'CN': CNAgent, 
    'CN0': CN0Agent, 
}

mkdir_p('output')
mkdir_p('output/logs')
mkdir_p('output/networks')
# mkdir_p('output/pred')
# mkdir_p('output/imgs')

parser = OptionParser()
parser.add_option('-a', '--agent', dest='agent', choices=['DCRAC', 'DCRACS', 'DCRACSE', 'DCRAC0', 'CN', 'CN0'], default='DCRACS')
parser.add_option('-n', '--net-type', dest='net_type', choices=['R', 'M', 'F'], default='R', help='Agent architecture type: [R]ecurrent, [M]emNN or [F]C')
parser.add_option('-r', '--replay', dest='replay', default='DER', choices=['STD', 'DER'], help='Replay type, one of "STD","DER"')
parser.add_option('-s', '--buffer-size', dest='buffer_size', default='10000', help='Replay buffer size', type=int)
parser.add_option('-m', '--memnn-size', dest='memnn_size', default='4', help='Memory network memory size', type=int)
parser.add_option('-d', '--dup', dest='dup', action='store_false', default=True, help='Extra training')
parser.add_option('-t', '--timesteps', dest='timesteps', default='4', help='Recurrent timesteps', type=int)
parser.add_option('-e', '--end_e', dest='end_e', default='0.01', help='Final epsilon value', type=float)
parser.add_option('-l', '--lr-c', dest='lr_c', default='0.02', help='Critic learning rate', type=float)
parser.add_option('-L', '--lr-a', dest='lr_a', default='0.25', help='Actor learning rate', type=float)
parser.add_option('--buffer-a', dest='buffer_a', default='2.', help='reply buffer error exponent', type=float)
parser.add_option('--buffer-e', dest='buffer_e', default='0.01', help='reply buffer error offset', type=float)
parser.add_option('-u', '--update-period', dest='updates', default='1', help='Update interval', type=int)
parser.add_option('-f', '--frame-skip', dest='frame_skip', default='4', help='Frame skip', type=int)
parser.add_option('-b', '--batch-size', dest='batch_size', default='64', help='Sample batch size', type=int)
parser.add_option('-g', '--discount', dest='discount', default='0.95', help='Discount factor', type=float)
parser.add_option('--anneal-steps', dest='steps', default='10000', help='steps',  type=int)
parser.add_option('-p', '--mode', dest='mode', choices=['regular', 'sparse'], default='regular')
parser.add_option('-v', '--obj-func', dest='obj_func', choices=['a', 'ap', 'td', 'q', 'y'], default='a')
parser.add_option('--no-action', dest='action_conc', action='store_false', default=True)
parser.add_option('--no-embd', dest='feature_embd', action='store_false', default=True)
parser.add_option('--gpu', dest='gpu_setting', choices=['1', '2', '3'], default='2', help='1 for CPU, 2 for GPU, 3 for CuDNN')
parser.add_option('--log-game', dest='log_game', action='store_true', default=False)
parser.add_option('--dst', dest='dst_view', choices=['3', '5'], default='5')

(options, args) = parser.parse_args()


for i in range(8):
    if i == 4: options.dst_view = '5'

    hyper_info = '{}_{}-r{}{}-d{}-t{}-batsiz{}-{}steps-lr{}-lr2{}-{}-acteval_{}'.format(
        options.agent, options.net_type, options.replay, str(options.buffer_size), options.dup, options.timesteps, 
        options.batch_size, options.steps, str(options.lr_c), str(options.lr_a), options.mode, options.obj_func)

    # create evironment
    env = DeepSeaTreasure(view=(5,5), scale=9) if options.dst_view == '5' else DeepSeaTreasure(view=(3,3), scale=15)

    all_weights = get_weights_from_json('./train_weights_dst.json') if options.mode == 'sparse' else get_weights_from_json('./train_weights_dst_r.json')
    timestamp = time.strftime('%m%d_%H%M', time.localtime())
    deep_agent = AGENT_DICT[options.agent]
    agent = deep_agent(env, 
                       gamma=options.discount,
                       weights=None,
                       timesteps=options.timesteps,
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
                       dup=options.dup,
                       action_conc=options.action_conc,
                       feature_embd=options.feature_embd,
                       extra='{}_{}'.format(timestamp, hyper_info),
                       gpu_setting=options.gpu_setting)

    steps_per_weight = 5000 if options.mode == 'sparse' else 1

    log_file_name = 'output/logs/{}_dst{}_rewards_{}.log'.format(timestamp, options.dst_view, hyper_info)
    with open(log_file_name, 'w', 1) as log_file:
        agent.train(log_file, options.steps, all_weights, steps_per_weight, options.steps*10, log_game_step=options.log_game)

    # print stats
    length, step_rewards, step_regrets, step_nb_episode = rebuild_log(total=options.steps*10, log_file=log_file_name)
    print_stats(length, step_rewards, step_regrets, step_nb_episode, write_to_file=True, timestamp=timestamp)




    # run test
    agent.use_prob = False
    with open('output/test1.log', 'w', 1) as test_log:
        all_weights = get_weights_from_json('./train_weights_dst.json')
        test_weights = all_weights[:20]
        agent.test(weights=test_weights, log_file=test_log)
    compute_log('output/test1.log', 'output/test1_result.csv')
    with open('output/test2.log', 'w', 1) as test_log:
        test_weights = np.array(list(zip(range(1,20), range(19,0,-1))))/20
        agent.test(weights=test_weights, log_file=test_log)
    compute_log('output/test2.log', 'output/test2_result.csv')

    if options.agent == 'CN':
        continue

    agent.use_prob = True
    agent.stoch_policy = False
    with open('output/test3.log', 'w', 1) as test_log:
        all_weights = get_weights_from_json('./train_weights_dst.json')
        test_weights = all_weights[:20]
        agent.test(weights=test_weights, log_file=test_log)
    compute_log('output/test3.log', 'output/test3_result.csv')
    with open('output/test4.log', 'w', 1) as test_log:
        test_weights = np.array(list(zip(range(1,20), range(19,0,-1))))/20
        agent.test(weights=test_weights, log_file=test_log)
    compute_log('output/test4.log', 'output/test4_result.csv')