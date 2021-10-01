import sys
import numpy as np



import matplotlib.pyplot as plt
def displayImage(image, transpose=False):
    fig = plt.imshow(image.transpose(1,0,2)) if transpose else plt.imshow(image) 
    plt.show()

if len(sys.argv) > 1 and int(sys.argv[1]) == 0:
    from minecart import Minecart
    # Generate minecart from configuration file (2 ores + 1 fuel objective, 5 mines)
    json_file = "mine_config.json"
    env = Minecart.from_json(json_file)
    # # Or alternatively, generate a random instance
    # env = Minecart(mine_cnt=5,ore_cnt=2,capacity=1)
else:
    from deep_sea_treasure import DeepSeaTreasure
    env = DeepSeaTreasure()

# Initial State
o_t = env.reset()
displayImage(o_t, False)
# print(o_t.shape)
# exit(0)

# flag indicates termination
terminal = False

while not terminal:
    # randomly pick an action
    # a_t = np.random.randint(env.action_space.shape[0])
    a_t = int(input())

    # apply picked action in the environment
    o_t1, r_t, terminal, s_t1 = env.step(a_t)

    # update state
    o_t = o_t1

    # displayImage(s_t1["pixels"])
    # displayImage(s_t1["pixels"], False)
    displayImage(o_t, False)
      
    print("Taking action", a_t, "with reward", r_t)
  
env.reset()
