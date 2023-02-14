import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import environment
import random
import os
import math
import copy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
gamma = 0.99  # discount factor


# random seed
# randomseed = 69963
# torch.manual_seed(randomseed)
# np.random.seed(randomseed)
# random.seed(randomseed)


def validation(algorithm):
    env = environment.environment(gamma, 'Valid', algorithm)  # instantiate the environment
    num_state = env.num_state  # the dimension of state space
    num_action = env.num_action  # the number of discretized actions
    num_agent = env.num_agent
    if algorithm in ['DQN', 'EASpace_DQN', 'Shaping']:
        class Net(nn.Module):
            # D3QN network
            def __init__(self):
                super(Net, self).__init__()
                self.layer1 = nn.Linear(num_state, 64)
                self.layer2 = nn.Linear(64, 64)
                self.layer3 = nn.Linear(64, 32)
                self.layer4 = nn.Linear(64, 32)
                self.layer5 = nn.Linear(32, num_action)
                self.layer6 = nn.Linear(32, 1)

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                advantage = torch.relu(self.layer3(x))
                advantage = self.layer5(advantage)
                # calculate state value
                state_value = torch.relu(self.layer4(x))
                state_value = self.layer6(state_value)
                # calculate Q value
                action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)
                return action_value
    if algorithm in ['Caps']:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layer1 = nn.Linear(num_state, 64)
                self.layer2 = nn.Linear(64, 64)
                self.layer3 = nn.Linear(64, 32)
                self.layer4 = nn.Linear(64, 32)
                self.layer5 = nn.Linear(32, num_action)
                self.layer6 = nn.Linear(32, 1)
                self.layer7 = nn.Linear(64, 32)
                self.layer8 = nn.Linear(32, 2)

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                advantage = torch.relu(self.layer3(x))
                advantage = self.layer5(advantage)
                # calculate state value
                state_value = torch.relu(self.layer4(x))
                state_value = self.layer6(state_value)
                # calculate Q value
                action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)
                termination = torch.relu(self.layer7(x))
                termination = torch.sigmoid(self.layer8(termination))
                return action_value, termination
    # calculate the success rate every 1000 episodes
    for file in ['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000']:
        # file = '1000'
        t_per_episode = list()
        is_collision = list()
        if algorithm in ['EASpace_DQN', 'DQN', 'Shaping', 'Caps']:
            net = Net()  # instantiate the network
            net.load_state_dict(torch.load(file + '.pt'))

        for num_episode in range(1000):
            state = env.reset()  # reset the environment
            while True:
                env.render()  # render the pursuit process
                # plt.savefig(str(j)) # save figures
                if algorithm in ['EASpace_DQN']:
                    action = np.zeros((1, 0))
                    # choose actions
                    for i in range(env.num_agent):
                        if env.action_chosen[0, i] > 24 and env.action_chosen[0, i] != 44:
                            action = np.hstack((action, env.action_chosen[:, i:i + 1] - 1))
                        else:
                            temp = state[:, i:i + 1].reshape(1, -1)
                            temp = net(torch.tensor(np.ravel(temp), dtype=torch.float32).view(1, -1))
                            action_temp = torch.max(temp, 1)[1].data.numpy()
                            action = np.hstack((action, np.array(action_temp, ndmin=2)))
                    action_temp = copy.deepcopy(action)
                if algorithm in ['Caps']:
                    action_total = np.zeros((1, 0))  # action buffer
                    # choose actions
                    for i in range(env.num_agent):
                        temp = torch.tensor(np.ravel(state[:, i]), dtype=torch.float32).view(1, -1)
                        actions_value, termination = net(temp)
                        if env.action_chosen[0, i] != -1 and (
                                env.action_chosen[0, i] == 24 or env.action_chosen[0, i] == 25):
                            if env.action_chosen[0, i] == 24:
                                if np.random.random() > termination[0, 0].detach().to('cpu').numpy():
                                    action = env.action_chosen[0, i]
                                else:
                                    action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()
                            if env.action_chosen[0, i] == 25:
                                if np.random.random() > termination[0, 1].detach().to('cpu').numpy():
                                    action = env.action_chosen[0, i]
                                else:
                                    action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()
                        else:
                            action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()
                        action_total = np.hstack((action_total, np.array([[action]])))
                    action_temp = copy.deepcopy(action_total)
                if algorithm in ['DQN', 'Shaping']:
                    action = np.zeros((1, 0))  # action buffer
                    # choose actions
                    for i in range(env.num_agent):
                        temp = state[:, i:i + 1].reshape(1, -1)
                        temp = net(torch.tensor(np.ravel(temp), dtype=torch.float32).view(1, -1))
                        action_temp = torch.max(temp, 1)[1].data.numpy()
                        action = np.hstack((action, np.array(action_temp, ndmin=2)))
                    action_temp = copy.deepcopy(action)
                # execute action
                state_next, reward, done = env.step(action_temp)
                temp1 = done == 1
                temp2 = done == 2
                temp = np.vstack((temp1, temp2))
                if np.any(done == 3):
                    # if there is one pursuer collides
                    t_per_episode.append(1000)
                    is_collision.append(1)
                    break
                if np.all(np.any(temp, axis=0, keepdims=True)):
                    # if all pursuers capture the evader or the episode reaches maximal length
                    t_per_episode.append(env.t)
                    is_collision.append(0)
                    break
                state = state_next
        np.savetxt(file + '_time.txt', t_per_episode)
        np.savetxt(file + '_collision.txt', is_collision)


if __name__ == '__main__':
    validation('EASpace_DQN')
