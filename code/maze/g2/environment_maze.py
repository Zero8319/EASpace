import numpy as np
from matplotlib import pyplot as plt
import copy
import torch
import torch.nn as nn

macro_action_length = 10


class Net(nn.Module):
    # DQN network
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class environment():
    def __init__(self, algorithm='DQN'):
        self.t = 0  # current timestep
        self.algorithm = algorithm
        self.num_row = 21  # the number of rows of the maze
        self.num_column = 24  # the number of columns of the maze
        self.num_state = 2  # the dimension of state space, x-y coordinates
        if self.algorithm == 'DQN':
            self.num_action = 4
        if self.algorithm == 'Caps':
            self.num_action = 8
            self.g1 = Net()
            self.g1.load_state_dict(torch.load('g1.pt'))
            self.g2 = Net()
            self.g2.load_state_dict(torch.load('g2.pt'))
            self.g3 = Net()
            self.g3.load_state_dict(torch.load('g3.pt'))
            self.g4 = Net()
            self.g4.load_state_dict(torch.load('g4.pt'))
        if self.algorithm == 'EASpace':
            self.num_action = 4 + macro_action_length * 4  # the number of actions in the integrated action space
            # 4 expert policies
            self.g1 = Net()
            self.g1.load_state_dict(torch.load('g1.pt'))
            self.g2 = Net()
            self.g2.load_state_dict(torch.load('g2.pt'))
            self.g3 = Net()
            self.g3.load_state_dict(torch.load('g3.pt'))
            self.g4 = Net()
            self.g4.load_state_dict(torch.load('g4.pt'))
        # obstacles
        self.obstacle0 = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        self.obstacle1 = np.array([[1, 1, 1, 1, 1], [0, 6, 12, 17, 23]])
        self.obstacle2 = np.array([[2, 2, 2, 2], [0, 6, 12, 23]])
        self.obstacle3 = np.array([[3, 3, 3, 3, 3], [0, 6, 12, 17, 23]])
        self.obstacle4 = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                   [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        self.obstacle5 = np.array([[5, 5, 5, 5], [0, 4, 19, 23]])
        self.obstacle6 = np.array([[6, 6, 6], [0, 4, 23]])
        self.obstacle7 = np.array([[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                                   [0, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 19, 23]])
        self.obstacle8 = np.array([[8, 8, 8, 8, 8, 8, 8, 8], [0, 7, 16, 19, 20, 21, 22, 23]])
        self.obstacle9 = np.array([[9, 9, 9, 9, 9, 9, 9, 9, 9], [0, 1, 2, 3, 4, 7, 16, 19, 23]])
        self.obstacle10 = np.array([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                    [0, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 23]])
        self.obstacle11 = np.array([[11, 11, 11, 11, 11, 11], [0, 7, 11, 16, 19, 23]])
        self.obstacle12 = np.array([[12, 12, 12, 12, 12, 12], [0, 4, 7, 11, 19, 23]])
        self.obstacle13 = np.array([[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
                                    [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 22, 23]])
        self.obstacle14 = np.array([[14, 14], [0, 23]])
        self.obstacle15 = np.array([[15, 15], [0, 23]])
        self.obstacle16 = np.array(
            [[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
             [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]])
        self.obstacle17 = np.array([[17, 17, 17, 17, 17], [0, 5, 12, 17, 23]])
        self.obstacle18 = np.array([[18, 18, 18, 18], [0, 5, 12, 23]])
        self.obstacle19 = np.array([[19, 19, 19, 19, 19], [0, 5, 12, 17, 23]])
        self.obstacle20 = np.array(
            [[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        self.obstacle = np.hstack(
            (self.obstacle0, self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle4, self.obstacle5,
             self.obstacle6, self.obstacle7, self.obstacle8, self.obstacle9, self.obstacle10,
             self.obstacle11, self.obstacle12, self.obstacle13, self.obstacle14, self.obstacle15,
             self.obstacle16, self.obstacle17, self.obstacle18, self.obstacle19, self.obstacle20))
        # the goals of source tasks and target tasks
        self.position_g1 = np.array([[2], [3]])
        self.position_g2 = np.array([[18], [3]])
        self.position_g3 = np.array([[18], [20]])
        self.position_g4 = np.array([[2], [20]])
        self.position_g = np.array([[11], [15]])
        self.position_gg = np.array([[3], [2]])
        # the current position of the agent
        self.agent_position = np.array([[0.], [0]])
        self.agent_position_last = np.array([[0.], [0]])
        # the current executed action
        self.action_chosen = -1

    def reset(self):
        self.t = 0
        # randomly select the initial position
        while True:
            temp1 = np.random.randint(self.num_row)
            temp2 = np.random.randint(self.num_column)
            temp = np.all(np.vstack((temp1 == self.obstacle[0:1, :], temp2 == self.obstacle[1:2, :])), axis=0)
            if not np.any(temp):
                break
        self.agent_position = np.array([[0.], [0]])
        self.agent_position[0, 0] = temp1
        self.agent_position[1, 0] = temp2
        self.agent_position_last = copy.deepcopy(self.agent_position)
        return self.agent_position

    def step(self, action, additional_reward=np.zeros((1, 3))):
        self.t += 1
        agent_position = copy.deepcopy(self.agent_position)
        self.agent_position_last = copy.deepcopy(self.agent_position)
        self.action_chosen = action
        if np.random.random() < 0.2:
            # the agent moves randomly with the probability of 0.2, while the selected action navigates the agent to the
            # desired direction with the probability of 0.8.
            action = np.random.randint(0, 4)
        if action == 0:
            # up
            agent_position[0, 0] -= 1 / 3
        if action == 1:
            # down
            agent_position[0, 0] += 1 / 3
        if action == 2:
            # left
            agent_position[1, 0] -= 1 / 3
        if action == 3:
            # right
            agent_position[1, 0] += 1 / 3
        if self.algorithm == 'Caps':
            if action > 3:
                if action == 4:
                    net = self.g1
                if action == 5:
                    net = self.g2
                if action == 6:
                    net = self.g3
                if action == 7:
                    net = self.g4
                temp = np.round(self.agent_position)
                actions_value = net(torch.FloatTensor(temp.reshape(1, -1)))
                action = torch.max(actions_value, 1)[1].data.numpy().item()
                if action == 0:
                    agent_position[0, 0] -= 1 / 3
                if action == 1:
                    agent_position[0, 0] += 1 / 3
                if action == 2:
                    agent_position[1, 0] -= 1 / 3
                if action == 3:
                    agent_position[1, 0] += 1 / 3
        if self.algorithm == 'EASpace':
            # execute expert policies
            if action > 3:
                if action < 4 + macro_action_length:
                    net = self.g1
                elif action < 4 + macro_action_length * 2:
                    net = self.g2
                elif action < 4 + macro_action_length * 3:
                    net = self.g3
                else:
                    net = self.g4
                temp = np.round(self.agent_position)
                actions_value = net(torch.FloatTensor(temp.reshape(1, -1)))
                action = torch.max(actions_value, 1)[1].data.numpy().item()
                if action == 0:
                    agent_position[0, 0] -= 1 / 3
                if action == 1:
                    agent_position[0, 0] += 1 / 3
                if action == 2:
                    agent_position[1, 0] -= 1 / 3
                if action == 3:
                    agent_position[1, 0] += 1 / 3
        # if there is obstacles in the desired gird, the agent does not move. Otherwise, it moves.
        temp = np.round(agent_position)
        temp1 = np.all(
            np.vstack((temp[0, 0] == self.obstacle[0:1, :], temp[1, 0] == self.obstacle[1:2, :])),
            axis=0)
        if not np.any(temp1):
            self.agent_position = agent_position

        if np.abs(self.agent_position[0, 0] - self.position_gg[0, 0]) < 0.55 and np.abs(
                self.agent_position[1, 0] - self.position_gg[1, 0]) < 0.55:
            # if the goal is reached
            reward = 10
            done = 1
        else:
            if self.t == 300:
                # if the maximal timestep is reached
                if np.sum(np.abs(self.agent_position_last - self.position_gg)) - np.sum(
                        np.abs(self.agent_position - self.position_gg)) < 0:
                    reward = -0.1
                elif np.sum(np.abs(self.agent_position_last - self.position_gg)) - np.sum(
                        np.abs(self.agent_position - self.position_gg)) > 0:
                    reward = 0.1
                else:
                    reward = 0
                done = 2
            else:
                if np.sum(np.abs(self.agent_position_last - self.position_gg)) - np.sum(
                        np.abs(self.agent_position - self.position_gg)) < 0:
                    reward = -0.1
                elif np.sum(np.abs(self.agent_position_last - self.position_gg)) - np.sum(
                        np.abs(self.agent_position - self.position_gg)) > 0:
                    reward = 0.1
                else:
                    reward = 0
                done = 0

        return self.agent_position, reward, done

    def render(self):
        plt.figure(1)
        plt.cla()
        table = np.zeros((self.num_row, self.num_column))
        table[self.obstacle[0, :], self.obstacle[1, :]] = 1
        agent_position = np.round(self.agent_position)
        table[int(agent_position[0, 0]), int(agent_position[1, 0])] = 0.5
        plt.imshow(table)
        plt.show(block=False)
        # plt.savefig(str(self.t))#whether save figures
        plt.pause(0.001)
