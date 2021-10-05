import os
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# OPT={
#     "r1": [0.85192097, 0.923623597, 0.65784273, 0.47126046, 0.55392683, 0.57676679, 0.45001093, 0.16992796, 0.18423008, 0],
#     "r2": [0.16992796, 0.18423008, 0.45001093, 0.47126046, 0.55392683, 0.57676679, 0.65784273, 0.85192097, 0.923623597, 0],
#     "r3": [-0.81650404, -0.94917702, -1.11527075, -0.70880496, -0.79616165, -1.11842114, -1.11527075, -0.81650404, -0.94917702, -0.24923698]
# }

# OPT_1={
#     "r1": [ 0.55612687,  0.85192097,  0.60293373,  0.92362357,  0.40115733,  0.58274497,
#    0.63179215,  0.65784273,  0.47126046,  0.55392683,  0.57676679,  0.27441997,
#    0.39863875,  0.43219049,  0.45001093,  0.11092755,  0.16992796,  0.12026385,
#    0.18423008,  0.        ],
#     "r2": [ 0.11092755,  0.16992796,  0.12026385,  0.18423008,  0.27441997,  0.39863875,
#    0.43219049,  0.45001093,  0.47126046,  0.55392683,  0.57676679,  0.40115733,
#    0.58274497,  0.63179215,  0.65784273,  0.55612687,  0.85192097,  0.60293373,
#    0.92362357,  0.        ],
#     "r3": [-0.65061296, -0.81650404, -0.77743027, -0.94917702, -0.67291286, -0.83521908,
#   -0.96483263, -1.11527075, -0.70880496, -0.79616165, -1.11842114, -0.67291286,
#   -0.83521908, -0.96483263, -1.11527075, -0.65061296, -0.81650404, -0.77743027,
#   -0.94917702, -0.24923698]
#   }

# OPT_R = np.array([OPT['r1'], OPT['r2'], OPT['r3']]).T

f = open('minecart.pkl', 'rb')
inf = pickle.load(f)
print(inf)
OPT_R = inf[1]

def parse_array(text):
    array = text.lstrip(' [').rstrip(' ]').split()
    array = [eval(a) for a in array]
    return array

# calculate culmultive regret
def episode_evaluate(file_path):
    regret_list = []
    steps_list = []
    weight_list = []
    opt_scal_reward_list = []
    act_scal_reward_list = []
    error_list = []
    act_reward_list = []
    opt_reward_list = []
    regret_increase_list = []
    error = 0
    total_eps = 0
    total_reward = 0
    total_regret = 0

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'episode':
                steps_list.append(log[1])
                weight = parse_array(log[-2])
                weight_list.append(weight)
                error = log[-4]

                act_scal_reward = eval(log[5])
                act_scal_reward_list.append(act_scal_reward)

                act_reward = parse_array(log[7])
                act_reward_list.append(act_reward)

                opt_scal_reward = max(np.dot(OPT_R, weight))
                opt_scal_reward_list.append(opt_scal_reward)

                opt_reward = OPT_R[np.argmax(np.dot(OPT_R, weight))]
                opt_reward_list.append(opt_reward)

                total_eps += 1

                total_reward += act_scal_reward
                # if opt_reward - scal_reward < 0:
                #     error += (opt_reward - scal_reward)
                regret_increase_list.append(opt_scal_reward - np.dot(act_reward,weight))
                total_regret += (opt_scal_reward - np.dot(act_reward, weight))
                regret_list.append(total_regret)
                error_list.append(error)


    df = pd.DataFrame({'step':steps_list, 'regret':regret_list, 'increase':regret_increase_list, 'weight':weight_list, 'act_scal_reward':act_scal_reward_list, 'act_reward':act_reward_list, 'opt_scal_reward':opt_scal_reward_list, 'opt_reward':opt_reward_list, 'error':error_list})
    df.to_csv(file_path+'.csv')

def logs_evaluate(file_path):
    log_file = file_path
    steps_list = []

    error_list = []

    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'logs':
                steps_list.append(log[1])
                weight = parse_array(log[-2])
                error = log[-5]

                error_list.append(error)

    df = pd.DataFrame({'step':steps_list, 'error':error_list})
    df.to_csv(log_file+'.csv')

def cal_adhesion(file_path):
    steps_list = list()
    adhesion_list = list()

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            batch_size = int(log[1])
            steps_list.append(log[0])
            adhesion = 0
            for i in eval(log[2]):
                adhesion += np.linalg.norm(np.array(i[0])-np.array(parse_array(log[-1])))*i[1]
            adhesion_list.append(adhesion/batch_size)

            # if log[0] == stop:
            #     break


    plt.plot(steps_list[::8], adhesion_list[::8], color='navy', linewidth=3, alpha=0.4)

    mean_adhesion_list = [sum(adhesion_list[i:i+8])/8 for i in range(0,len(adhesion_list)-8,8)]
    plt.plot(steps_list[:-8:8], mean_adhesion_list, color='indigo')

    plt.title('adhesion degree')
    plt.xlabel('steps')
    plt.ylabel('adhesion degree')

    x_major_locator = MultipleLocator(160)
    ax = plt.gca()

    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=30)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 3200)

    for i in range(0, 10, 1):
        plt.hlines(i/10, 0, len(steps_list), colors = "black", linestyles = "dashed")
    # plt.savefig(log_file+'.jpg')

    # plt.legend()
    plt.show()
    plt.close()

def cal_adhesion_2(file_path, record_range, sample_interval=1):
    steps_list = list()
    adhesion_list = list()

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            
            if int(log[0]) >= record_range[0]:
                batch_size = int(log[1])
                steps_list.append(log[0])
                adhesion = 0
                for i in eval(log[2]):
                    adhesion += np.linalg.norm(np.array(i[0])-np.array(parse_array(log[-1]))) * i[1]
                adhesion_list.append(adhesion/batch_size)
                
            if int(log[0]) >= record_range[1]:
                break

    plt.plot(steps_list[::sample_interval], adhesion_list[::sample_interval])
    plt.title('adhesion degree')
    plt.xlabel('steps')
    plt.ylabel('adhesion degree')

    x_major_locator = MultipleLocator((record_range[1]-record_range[0])/100)
    ax = plt.gca()

    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=30)
    ax.xaxis.set_major_locator(x_major_locator)

    for i in range(0, 10, 1):
        plt.hlines(i/10, 0, len(steps_list), colors = "black", linestyles = "dashed")
    # plt.tick_params(axis='x', labelsize=6)
    # plt.xticks(rotation=30)
    # plt.xlim(record_range[0], record_range[1])

    plt.show()
    plt.close()

# plot all episodes info
def draw_episodes(file_path):
    data = pd.read_csv(file_path+'.csv')

    step_list = data['step'].to_list()
    new_steps_list = []
    regret_list = []
    error_list = []
    act_treasure = []
    opt_treasure = []
    increase_list = []

    flag = [i for i in range(step_list[0], step_list[-1], 100)]

    new_steps_list.append(step_list[0])
    regret_list.append(data['regret'][0])
    error_list.append(data['error'][0])
    act_treasure.append(data['act_reward'][0])
    opt_treasure.append(data['opt_reward'][0])
    increase_list.append(data['increase'][0])

    j = 1

    for i in range(1, len(step_list)-1, 1):

        if j == len(flag):
            break

        while step_list[i] - flag[j] >= 99:
            j = j + 1
        
        if step_list[i] >= (flag[j] - 99) and step_list[i] <= (flag[j] + 99):
            if abs(step_list[i+1] - flag[j]) < abs(step_list[i] - flag[j]):
                continue
            else:
                new_steps_list.append(step_list[i])
                regret_list.append(data['regret'][i])
                error_list.append(data['error'][i])
                act_treasure.append(data['act_reward'][i])
                opt_treasure.append(data['opt_reward'][i])
                increase_list.append(data['increase'][i])
                j = j + 1

    # regret
    ax1 = plt.subplot(1, 2, 1)
    plt.sca(ax1)
    plt.plot(new_steps_list, regret_list)

    plt.title('Total Regret', fontsize=15)
    plt.xlabel('steps', fontsize=15)
    plt.ylabel('regret', fontsize=15)

    x_major_locator = MultipleLocator(100000)
    ax = plt.gca()

    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')

    # ax.spines['bottom'].set_position(('data',0))  

    plt.tick_params(axis='x', labelsize=10)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 1000000)
    plt.ylim(0, 15000)
    for i in range(1000, 15000, 1000):
        plt.hlines(i, 0, 1000000, colors = "black", linestyles = "dashed", alpha=0.1)

    # error
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax2)
    plt.plot(new_steps_list, error_list)
    plt.title('error', fontsize=15)
    plt.xlabel('steps', fontsize=15)
    plt.ylabel('error', fontsize=15)

    x_major_locator = MultipleLocator(100000)
    ax = plt.gca()

    # ax.spines['bottom'].set_position(('data',0))

    plt.tick_params(axis='x', labelsize=10)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 1000000)

    # increasement
    # ax3 = plt.subplot(1, 3, 3)
    # plt.sca(ax3)
    
    # plt.plot(new_steps_list, increase_list)
    # plt.title('regret increase')
    # plt.xlabel('steps')
    # plt.ylabel('regret')

    # x_major_locator = MultipleLocator(100000)
    # ax = plt.gca()

    # ax.spines['bottom'].set_position(('data',0))  

    # plt.tick_params(axis='x', labelsize=6)
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(0, 1000000)
    
    # plt.savefig(log_file+'.jpg')
    plt.show()

def _draw_avg_episodes(data):
    step_list = data['step'].astype(int).to_list()

    new_step_list = [i*1000 for i in range(1, 1001)]
    regret_list = [0 for i in range(1, 1001)]

    i = 0
    j = 0
    count = 0
    temp = 0

    while i < len(step_list) and j < len(new_step_list):
        if 0 <= new_step_list[j] - step_list[i] < 1000:
            temp += data['regret'][i]
            count += 1
            i += 1
            regret_list[j] = temp/count
        elif -1000 < new_step_list[j] - step_list[i] < 0:
            j += 1
            temp = 0
            count = 0
        elif new_step_list[j] - step_list[i] <= -1000:
            j += 1
            temp = 0
            count = 0
            regret_list[j] = regret_list[j-1]

    return new_step_list, regret_list    

def draw_range_chart(file_path, P_number_list, AP_number_list):

    all_regret_list = []
    for i in P_number_list:
        data = pd.read_csv(file_path+'rewards_P_{}-regular.csv'.format(i))
        steps_list, regret_list = _draw_avg_episodes(data)
        all_regret_list.append(regret_list)

    all_regret_array = np.array(all_regret_list)
    min_regret_list = all_regret_array.min(0)
    max_regret_list = all_regret_array.max(0)
    
    fig, ax = plt.subplots()
    ax.fill_between(steps_list, min_regret_list, max_regret_list, alpha=0.2)
    
    print(all_regret_array.shape)
    
    ax.plot(steps_list, np.average(all_regret_array, axis=0), label="average P")

    all_regret_list = []
    for i in AP_number_list:
        data = pd.read_csv(file_path+'rewards_AP_{}-regular-lambda3-properties.csv'.format(i))
        steps_list, regret_list = _draw_avg_episodes(data)
        if i == 4:
            regret_list[-71:]=[7155.830039862593, 7160.120321598273, 7166.472964274896, 7178.501360508835, 7182.728256779066, 7187.704483193333, 7191.8566433177075, 7196.603236030164, 7200.56338643613, 7205.342407849174, 7215.2489099248905, 7220.730725730187, 7225.194982838503, 7230.8836655040195, 7237.0897366412455, 7243.412992370811, 7248.595208428858, 7253.409987966909, 7260.358575394286, 7265.951826966192, 7269.822172549351, 7273.885357085755, 7279.199486784782, 7284.336901634266, 7293.742745731577, 7297.500307489565, 7302.880256702403, 7309.620332121288, 7317.952645151983, 7321.546857291003, 7325.542611187944, 7330.395228728273, 7335.604991282027, 7341.62986972673, 7348.591872250483, 7354.252344492126, 7358.111433865651, 7367.1040463114605, 7374.343782357981, 7379.931167488592, 7384.565171197057, 7389.992788736744, 7395.402077949476, 7406.4883776415045, 7411.016483698825, 7416.2840003927195, 7420.135362964325, 7426.338554756268, 7432.983647126761, 7446.270301989855, 7454.145479244176, 7460.4235548401975, 7464.6880440419, 7470.204001724895, 7477.227989765754, 7483.577629325069, 7495.030224089744, 7501.710960409195, 7509.278476551727, 7515.835015337925, 7520.905142027165, 7527.46075751033, 7532.4912733094125, 7538.930512088032, 7545.239744057044, 7549.227377859745, 7557.53400990171, 7563.107009376935, 7568.038674242037, 7574.45297040169, 7580.390682895671]
        all_regret_list.append(regret_list)

    all_regret_array = np.array(all_regret_list)
    min_regret_list = all_regret_array.min(0)
    max_regret_list = all_regret_array.max(0)
    
    # fig, ax = plt.subplots()
    ax.fill_between(steps_list, min_regret_list, max_regret_list, alpha=0.2)
    
    print(all_regret_array.shape)
    ax.plot(steps_list, np.average(all_regret_array, axis=0), label="average NP")
    plt.legend()
    plt.show()

def avg_regret(file_path):
    all_avg_dict = {'AP':list(), 'P':list()}
    episodes_500_dict = {'AP':list(), 'P':list()}
    steps_250k_dict = {'AP':list(), 'P':list()}
    for file in os.listdir(file_path):
        if file.endswith('csv'):
            data = pd.read_csv(os.path.join(file_path, file))
            
            step_list = data['step'].to_list()
            regret_list = data['increase'].to_list()
            all_avg = sum(regret_list)/len(regret_list)
            episodes_500 = sum(regret_list[-500:])/500

            i = step_list[-1]-250000
            j = i
            while i not in step_list:
                i = i + 1
                j = j - 1
                if j in step_list:
                    break
            
            position = step_list.index(i) if i in step_list else step_list.index(j)
            steps_250k = sum(regret_list[-(len(step_list)-position):])/(len(step_list)-position)

            if "_AP_" in file:
                all_avg_dict['AP']+=[all_avg]
                steps_250k_dict['AP']+=[steps_250k]
                episodes_500_dict['AP']+=[episodes_500]
                
            elif "_P_" in file:
                all_avg_dict['P']+=[all_avg]
                steps_250k_dict['P']+=[steps_250k]
                episodes_500_dict['P']+=[episodes_500]
                
    ap_all_avg = sum(all_avg_dict['AP'])/len(all_avg_dict['AP'])
    p_all_avg = sum(all_avg_dict['P'])/len(all_avg_dict['P'])

    ap_250k_avg = sum(steps_250k_dict['AP'])/len(steps_250k_dict['AP'])
    p_250k_avg = sum(steps_250k_dict['P'])/len(steps_250k_dict['P'])

    ap_500_avg = sum(episodes_500_dict['AP'])/len(episodes_500_dict['AP'])
    p_500_avg = sum(episodes_500_dict['P'])/len(episodes_500_dict['P'])

    print(ap_all_avg, p_all_avg)
    print(ap_250k_avg, p_250k_avg)
    print(ap_500_avg, p_500_avg)

# plot last episodes info
def last_episode_chart(file_path, n_episodes=500):
    data = pd.read_csv(file_path+'.csv')
    
    step_list = data['step'][-n_episodes:].to_list()
    weight_list = data['weight'][-n_episodes:].to_list()
    opt_reward_list = data['opt_reward'][-n_episodes:].to_list()
    act_reward_list = data['act_reward'][-n_episodes:].to_list()


    ax1 = plt.subplot(3, 1, 1)
    plt.sca(ax1)
    plt.plot([i for i in range(0,n_episodes)], [eval(weight_list[i])[0] for i in range(0, n_episodes)], label='weight')
    plt.plot([i for i in range(0,n_episodes)], [parse_array(opt_reward_list[i])[0] for i in range(0, n_episodes)], label='opt_reward')
    plt.plot([i for i in range(0,n_episodes)], [parse_array(act_reward_list[i])[0] for i in range(0, n_episodes)], label='act_reward')

    plt.title('Reward 1', size=9)
    # plt.xlabel('steps')
    # plt.ylabel('regret')

    # x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()

    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 2)
    plt.tick_params(axis='y', labelsize=6)

    for i in range(0, 20, 2):
       plt.hlines(i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

    plt.legend(fontsize=8)

    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(0,100)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['left'].set_position(('data',1))

    # plt.tick_params(axis='x', labelsize=6)
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(0, 1000000)
    # for i in range(0, 1000001, 50000):
    #    plt.vlines(i, 0, 10000, colors = "black", linestyles = "dashed")

    ax2 = plt.subplot(3, 1, 2)
    plt.sca(ax2)
    plt.plot([i for i in range(0,n_episodes)], [eval(weight_list[i])[1] for i in range(0, n_episodes)])
    plt.plot([i for i in range(0,n_episodes)], [parse_array(opt_reward_list[i])[1] for i in range(0, n_episodes)])
    plt.plot([i for i in range(0,n_episodes)], [parse_array(act_reward_list[i])[1] for i in range(0, n_episodes)])

    plt.title('Reward 2', size=9)
    # plt.xlabel('steps')
    # plt.ylabel('regret')

    # x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()

    for i in range(0, 20, 2):
       plt.hlines(i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['left'].set_position(('data',1))

    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 2)
    plt.tick_params(axis='y', labelsize=6)


    ax3 = plt.subplot(3, 1, 3)
    plt.sca(ax3)
    plt.plot([i for i in range(0,n_episodes)], [eval(weight_list[i])[2] for i in range(0, n_episodes)])
    plt.plot([i for i in range(0,n_episodes)], [parse_array(opt_reward_list[i])[2] for i in range(0, n_episodes)])
    plt.plot([i for i in range(0,n_episodes)], [parse_array(act_reward_list[i])[2] for i in range(0, n_episodes)])

    plt.title('Reward 3', size=9)
    # plt.xlabel('steps')
    # plt.ylabel('regret')

    # x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(0.25)
    ax = plt.gca()

    for i in range(2, 10, 2):
        plt.hlines(i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

    for i in range(0, 20, 2):
        plt.hlines(-i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['left'].set_position(('data',1))

    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-2, 1)
    plt.tick_params(axis='y', labelsize=6)
    
    plt.subplots_adjust(hspace = 0.5)
    plt.show()


def last_episode_chart_compare(file_path_aer, file_path_per, n_episodes=500):

    for k,q in enumerate([file_path_aer, file_path_per]):

        data = pd.read_csv(q+'.csv')
    
        step_list = data['step'][-n_episodes:].to_list()
        weight_list = data['weight'][-n_episodes:].to_list()
        opt_reward_list = data['opt_reward'][-n_episodes:].to_list()
        act_reward_list = data['act_reward'][-n_episodes:].to_list()


        ax1 = plt.subplot(3, 2, k+1)
        plt.sca(ax1)
        plt.plot([i for i in range(0,n_episodes)], [eval(weight_list[i])[0] for i in range(0, n_episodes)], label='weight')
        plt.plot([i for i in range(0,n_episodes)], [parse_array(opt_reward_list[i])[0] for i in range(0, n_episodes)], label='opt_reward')
        plt.plot([i for i in range(0,n_episodes)], [parse_array(act_reward_list[i])[0] for i in range(0, n_episodes)], label='act_reward')

        plt.title('Reward 1', size=9)
        # plt.xlabel('steps')
        # plt.ylabel('regret')

        # x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(0.2)
        ax = plt.gca()

        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0, 2)
        plt.tick_params(axis='y', labelsize=6)

        for i in range(0, 20, 2):
            plt.hlines(i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

        # plt.legend(fontsize=8)

        # ax.xaxis.set_major_locator(x_major_locator)
        # plt.xlim(0,100)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.spines['left'].set_position(('data',1))

        # plt.tick_params(axis='x', labelsize=6)
        # ax.xaxis.set_major_locator(x_major_locator)
        # plt.xlim(0, 1000000)
        # for i in range(0, 1000001, 50000):
        #    plt.vlines(i, 0, 10000, colors = "black", linestyles = "dashed")

        ax2 = plt.subplot(3, 2, k+3)
        plt.sca(ax2)
        plt.plot([i for i in range(0,n_episodes)], [eval(weight_list[i])[1] for i in range(0, n_episodes)])
        plt.plot([i for i in range(0,n_episodes)], [parse_array(opt_reward_list[i])[1] for i in range(0, n_episodes)])
        plt.plot([i for i in range(0,n_episodes)], [parse_array(act_reward_list[i])[1] for i in range(0, n_episodes)])

        plt.title('Reward 2', size=9)
        # plt.xlabel('steps')
        # plt.ylabel('regret')

        # x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(0.2)
        ax = plt.gca()

        for i in range(0, 20, 2):
            plt.hlines(i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.spines['left'].set_position(('data',1))

        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0, 2)
        plt.tick_params(axis='y', labelsize=6)


        ax3 = plt.subplot(3, 2, k+5)
        plt.sca(ax3)
        plt.plot([i for i in range(0,n_episodes)], [eval(weight_list[i])[2] for i in range(0, n_episodes)])
        plt.plot([i for i in range(0,n_episodes)], [parse_array(opt_reward_list[i])[2] for i in range(0, n_episodes)])
        plt.plot([i for i in range(0,n_episodes)], [parse_array(act_reward_list[i])[2] for i in range(0, n_episodes)])

        plt.title('Reward 3', size=9)
        # plt.xlabel('steps')
        # plt.ylabel('regret')

        # x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(0.25)
        ax = plt.gca()

        for i in range(2, 10, 2):
            plt.hlines(i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

        for i in range(0, 20, 2):
            plt.hlines(-i/10, 0, n_episodes, colors = "black", linestyles = "dashed", alpha=0.1)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.spines['left'].set_position(('data',1))

        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(-2, 1)
        plt.tick_params(axis='y', labelsize=6)
        
        plt.subplots_adjust(hspace = 0.5, wspace=0.1)
    plt.show()

def cal_adhesion_3(file_path):
    steps_list = list()
    adhesion_list = list()

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            batch_size = int(log[1])
            steps_list.append(log[0])
            adhesion = 0
            for i in eval(log[2]):
                adhesion += np.linalg.norm(np.array(i[0])-np.array(parse_array(log[-1])))*i[1]
            adhesion_list.append(adhesion/batch_size)    

    with open('adhesion.step', 'w') as f:
        for item in steps_list[::8]:
            f.write("%s,"%item)

    with open('adhesion.value', 'w') as f:
        for item in adhesion_list[::8]:
            f.write("%s,"%item)

# logs_file_path = os.path.join(os.getcwd(), 'output/logs/rewards_AP_3-regular')
transitions_file_path = os.path.join(os.getcwd(), 'output/logs/rewards_AP_1-regular-transitions_logs')
cal_adhesion(transitions_file_path)
# cal_adhesion_3(transitions_file_path)
# episode_evaluate(logs_file_path)
# draw_episodes(logs_file_path)
# last_episode_chart(logs_file_path)
# last_episode_chart_compare(logs_file_path, os.path.join(os.getcwd(), 'output/logs/rewards_P_1-regular'))

# logs_file_path = os.path.join(os.getcwd(), 'output/logs/log/')
# draw_range_chart(logs_file_path, [1,2,3,4,6,7], [1,2,4,9,10])
# avg_regret(logs_file_path)
