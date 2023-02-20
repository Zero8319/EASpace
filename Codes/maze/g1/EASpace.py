import torch
import torch.nn as nn
import numpy as np
import environment_maze
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
import random
import os
from validation import validation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Hyper Parameters
episode_max = 8000  # the amount of episodes used to train
batch_size = 128  # batch size
lr = 7e-5  # learning rate
epsilon_origin = 1  # original epsilon
epsilon_decrement = 1 / 4000  # epsilon decay
gamma = 0.99  # the discount factor
target_replace_iter = 500  # update frequency of target network
memory_size = int(1e6)  # the size of replay memory
env = environment_maze.environment('EASpace')  # instantiate the environment

num_action = env.num_action  # the number of actions in the integrated action space
num_state = env.num_state  # the dimension of state space

option_bonus = 0.01  # macro action bonus c
macro_action_length = 10  # maximal length of macro actions tau_0


# random seed
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)


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


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # evaluation network and target network
        self.learn_step_counter = 0  # the counter of update
        self.memory = np.zeros((memory_size, num_state * 2 + 3))  # replay memory
        self.memory_counter = 0  # the counter of replay memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)  # adam optimizer
        self.loss_func = nn.MSELoss()  # MSE loss

    def choose_action(self, state, epsilon):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if env.action_chosen > 3 and env.action_chosen not in [4, 4 + macro_action_length, 4 + macro_action_length * 2,
                                                               4 + macro_action_length * 3]:
            # if the macro action continues, output the macro action derived from the same expert policy but lasts for
            # tau-1 timesteps, where tau is the duration of the last-timestep macro action
            action = env.action_chosen - 1
        else:
            # if the macro action terminates
            if np.random.uniform() > epsilon:
                # select the macro action with maximal action value
                actions_value = self.eval_net(state)
                action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

            else:
                if np.random.random() > 0.2:
                    # choose the primitive actions randomly. Since DQN is off-policy, the probability of selecting
                    # primitive actions (0.2 here) can be adjusted arbitrarily. If the expert policies are generally
                    # helpful to the target task, the probability of selecting primitive actions should be little, and
                    # vice versa.
                    action = np.random.randint(0, 4)
                else:
                    # choose the long-duration macro actions randomly. Here we only select the longest macro actions,
                    # then store different-length macro actions into the replay buffer.
                    action = np.random.choice(
                        [3 + macro_action_length, 3 + macro_action_length * 2, 3 + macro_action_length * 3,
                         3 + macro_action_length * 4])
        return action

    def store_transition(self, transition):
        self.memory[self.memory_counter % memory_size:self.memory_counter % memory_size + 1, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % target_replace_iter == 0:
            # periodically update the target network
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample transition batch
        index = np.random.randint(memory_size, size=(batch_size,))
        b_memory = self.memory[index, :]
        b_state = torch.FloatTensor(b_memory[:, :num_state]).to(device)
        b_action = torch.LongTensor(b_memory[:, num_state:num_state + 1]).to(device)
        b_reward = torch.FloatTensor(b_memory[:, num_state + 1:num_state + 2]).to(device)
        b_state_next = torch.FloatTensor(b_memory[:, num_state + 2:num_state * 2 + 2]).to(device)
        b_done = torch.FloatTensor(b_memory[:, num_state * 2 + 2:num_state * 2 + 3]).to(device)

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next_targetnet = self.target_net(b_state_next)
        q_next_evalnet = self.eval_net(b_state_next)
        q_next = q_next_targetnet.gather(1, torch.argmax(q_next_evalnet, axis=1, keepdim=True))
        # IMARL
        temp = self.target_net(b_state_next).gather(1, torch.abs(b_action - 1))
        index = torch.all(torch.hstack((b_action > 3, b_action != 4, b_action != 4 + macro_action_length,
                                        b_action != 4 + macro_action_length * 2,
                                        b_action != 4 + macro_action_length * 3)),
                          axis=1)
        q_next[index, :] = temp[index, :]
        q_target = b_reward + gamma * torch.abs(1 - b_done) * q_next
        # calculate loss
        loss = self.loss_func(q_eval, q_target.detach())
        # back propagate
        self.optimizer.zero_grad()
        loss.backward()
        # update parameters
        self.optimizer.step()


dqn = DQN()
device = torch.device('cuda')
dqn.eval_net.to(device)
dqn.target_net.to(device)

i_episode = 0

episode_return_total = np.zeros(0)  # episode return recorder

while True:
    state = env.reset()  # reset the environment
    episode_return = 0  # cumulative reward
    while True:
        action = dqn.choose_action(state, max(epsilon_origin - epsilon_decrement * i_episode, 0.01))
        # execute actions
        state_next, reward, done = env.step(action)
        # env.render()  # render the training process
        episode_return += reward
        if action > 3 and np.any(state != state_next):
            # combine the task reward and the macro action bonus
            reward += option_bonus * ((action - 4) % macro_action_length)
        transition = np.hstack((state.reshape(1, -1), np.array(action).reshape(1, -1), np.array(reward).reshape(1, -1),
                                state_next.reshape(1, -1), np.array(done).reshape(1, -1)))
        # Store the transitions into the reply memory. For simplicity, we only store tau transitions during the
        # execution of a tau-timestep macro action. But it is obvious we can store all macro actions that are derived
        # from the same expert policy at each timestep, which means that totally tau*tau0 transitions we acquire during
        # the this tau timesteps.
        dqn.store_transition(transition)
        if done:
            if dqn.memory_counter > memory_size:  # if the replay memory collects enough transitions
                for _ in range(300):
                    dqn.learn()  # train the network
                i_episode += 1
            break
        state = state_next
    # if the replay memory doesn't collect enough transitions, print information
    if dqn.memory_counter < memory_size:
        temp = "collecting experiences: " + str(dqn.memory_counter) + ' / ' + str(memory_size)
        print(temp)
    # if the replay memory collects enough transitions, plot cumulative reward and print information
    if dqn.memory_counter >= memory_size:
        episode_return_total = np.hstack((episode_return_total, episode_return))
        print('i_episode: ', i_episode, 'episode_return: ', round(episode_return, 2))
    # periodically save networks
    if i_episode % 1000 == 0:
        net = dqn.eval_net
        string = str(i_episode) + '.pt'
        torch.save(net.state_dict(), string)
        string = str(i_episode) + '.txt'
        np.savetxt(string, episode_return_total)
    # kill the training process, then calculate the success rate of all saved policies
    if i_episode == episode_max:
        validation('EASpace')
        break
