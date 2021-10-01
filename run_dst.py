import numpy as np
from deep_sea_treasure import DeepSeaTreasure
from agentT import DCRACSRunAgent, CNRunAgent
from utils import get_weights_from_json

all_weights = get_weights_from_json('./train_weights_dst.json')
test_weights = all_weights[:20]
# test_weights = np.array(list(zip(range(1,10), range(9,0,-1))))/10

log_file_name = 'output/test.log'
log_file = open(log_file_name, 'w', 1)
env = DeepSeaTreasure(view=(5,5), scale=9)

path='output/networks/model_name.h5'
agent = CNRunAgent(env=env, path=path)
agent.test(weights=test_weights, log_file=log_file)
