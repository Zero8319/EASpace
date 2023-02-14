import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import environment_maze
import random
import os
import math
import copy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random seed
# randomseed = 69963
# torch.manual_seed(randomseed)
# np.random.seed(randomseed)
# random.seed(randomseed)

macro_action_length = 10  # the maximal length of macro actions tau0


def validation(algorithm):
    env = environment_maze.environment(algorithm)  # instantiate the environment
    num_state = env.num_state  # the dimension of state space
    num_action = env.num_action  # the number of discretized actions
    if algorithm in ['Caps']:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layer1 = nn.Linear(num_state, 64)
                self.layer2 = nn.Linear(64, 64)
                self.layer3 = nn.Linear(64, num_action)
                self.layer4 = nn.Linear(64, 4)

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                actions_value = self.layer3(x)
                termination = torch.sigmoid(self.layer4(x))
                return actions_value, termination

        class Net2(nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.layer1 = nn.Linear(2, 64)
                self.layer2 = nn.Linear(64, 64)
                self.layer3 = nn.Linear(64, 4)

            def forward(self, x):
                '''
                Input:
                    x: observations
                output:
                    action_value: Q value for each action
                '''
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                x = self.layer3(x)
                return x

        if algorithm in ['EASpace', 'Caps']:
            g1 = Net2()
            g1.load_state_dict(torch.load('g1.pt'))
            g2 = Net2()
            g2.load_state_dict(torch.load('g2.pt'))
            g3 = Net2()
            g3.load_state_dict(torch.load('g3.pt'))
            g4 = Net2()
            g4.load_state_dict(torch.load('g4.pt'))
    if algorithm in ['EASpace', 'DQN']:
        class Net(nn.Module):
            # DQN network
            def __init__(self):
                super(Net, self).__init__()
                self.layer1 = nn.Linear(num_state, 64)
                self.layer2 = nn.Linear(64, 64)
                self.layer3 = nn.Linear(64, num_action)

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                x = self.layer3(x)
                return x
    # calculate the success rate every 1000 episodes
    for file in ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000']:
        # file = '1000'
        time = list()
        if algorithm in ['DQN', 'Caps', 'EASpace']:
            net = Net()  # instantiate the network
            net.load_state_dict(torch.load(file + '.pt'))

        for num_episode in range(1000):
            state = env.reset()  # reset the environment
            while True:
                env.render()  # render the pursuit process
                if algorithm in ['DQN']:
                    actions_value = net(torch.FloatTensor(state.reshape(1, -1)))
                    action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()
                if algorithm in ['Caps']:
                    actions_value, termination = net(torch.FloatTensor(state.reshape(1, -1)))
                    # epsilon-greedy method
                    if env.action_chosen > 3:
                        if np.random.random() < termination[0, env.action_chosen - 4].detach().cpu().numpy().item():
                            action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

                        else:
                            action = env.action_chosen
                    else:
                        action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()
                if algorithm in ['EASpace']:
                    state = torch.FloatTensor(state.reshape(1, -1))
                    # epsilon-greedy method
                    if env.action_chosen > 3 and env.action_chosen not in [4, 4 + macro_action_length,
                                                                           4 + macro_action_length * 2,
                                                                           4 + macro_action_length * 3]:
                        action = env.action_chosen - 1
                    else:
                        actions_value = net(state)
                        action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

                # execute action
                # action=19
                state_next, reward, done = env.step(action)
                if done:
                    # record the timesteps. If it is less than 300, the agent finds the goal successfully
                    time.append(env.t)
                    break
                state = state_next
        np.savetxt(file + '_time.txt', time)


if __name__ == '__main__':
    validation('EASpace')
