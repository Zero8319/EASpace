import torch
import torch.nn as nn
import numpy as np
import environment
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
import random
from prioritized_memory import Memory
from validation import validation
import os

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
env = environment.environment(gamma, 'Train', 'EASpace_DQN')  # instantiate the environment

num_action = env.num_action  # the number of actions in the integrated action space
num_state = env.num_state  # the dimension of state space
num_agent = env.num_agent  # the number of pursuers

option_bonus = 0.01  # macro action bonus c


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


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # evaluation network and target network
        self.learn_step_counter = 0  # the counter of update
        self.memory = Memory(memory_size, env.num_state, 1, env.num_agent)  # the prioritized replay memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)  # adam optimizer
        self.loss_func = nn.MSELoss(reduction='none')  # MSE loss
        self.max_td_error = 0.  # the maximal td error

    def choose_action(self, state, epsilon, action_chosen):
        if env.action_chosen[0, i] > 24 and env.action_chosen[0, i] != 44:
            # if the macro action continues, output the macro action derived from the same expert policy but lasts for
            # tau-1 timesteps, where tau is the duration of the last-timestep macro action
            action = action_chosen - 1
        else:
            # if the macro action terminates
            state = torch.tensor(np.ravel(state), dtype=torch.float32, device=device).view(1, -1)
            # epsilon-greedy method
            if np.random.uniform() > epsilon:
                # select the macro action with maximal action value
                actions_value = self.eval_net(state)
                action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

            else:
                # choose among the primitive actions and macro actions randomly. Since D3QN is off-policy, here we only
                # select the longest macro actions, then store different-length macro actions into the replay buffer.
                action = np.random.choice(np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 43, 63]))
        return action

    def store_transition(self, transition):
        transition = np.ravel(transition)
        self.memory.add(self.max_td_error, transition)

    def learn(self, i_episode):
        if self.learn_step_counter % target_replace_iter == 0:
            # periodically update the target network
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample transition batch
        b_memory, indexs, omega = self.memory.sample(batch_size, i_episode, episode_max)
        b_state = torch.tensor(b_memory[:, :num_state], dtype=torch.float32, device=device)
        b_action = torch.tensor(b_memory[:, num_state:num_state + 1], dtype=torch.int64, device=device)
        b_reward = torch.tensor(b_memory[:, num_state + 1:num_state + 2], dtype=torch.float32, device=device)
        b_state_next = torch.tensor(b_memory[:, num_state + 2:num_state * 2 + 2], dtype=torch.float32, device=device)
        b_done = torch.tensor(b_memory[:, num_state * 2 + 2:num_state * 2 + 3], dtype=torch.float32, device=device)

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next_targetnet = self.target_net(b_state_next)
        q_next_evalnet = self.eval_net(b_state_next)
        q_next = q_next_targetnet.gather(1, torch.argmax(q_next_evalnet, axis=1, keepdim=True))
        # IMARL
        temp = self.target_net(b_state_next).gather(1, torch.abs(b_action - 1))
        index = torch.all(torch.hstack((b_action > 24, b_action != 44)), axis=1)
        q_next[index, :] = temp[index, :]
        q_target = b_reward + gamma * torch.abs(1 - b_done) * q_next
        # calculate td errors
        td_errors = (q_target - q_eval).to('cpu').detach().numpy().reshape((-1, 1))
        # update prioritized replay memory
        self.max_td_error = max(np.max(np.abs(td_errors)), self.max_td_error)
        for i in range(batch_size):
            index = indexs[i, 0]
            td_error = td_errors[i, 0]
            self.memory.update(index, td_error)
        # calculate loss
        loss = (self.loss_func(q_eval, q_target.detach()) * torch.FloatTensor(omega).to(device).detach()).mean()
        # back propagate
        self.optimizer.zero_grad()
        loss.backward()
        # update parameters
        self.optimizer.step()


class RunningStat:
    # Calculate the mean and std of all previous rewards.
    def __init__(self):
        self.n = 0  # the number of reward signals collected
        self.mean = np.zeros((1,))  # the mean of all rewards
        self.s = np.zeros((1,))
        self.std = np.zeros((1,))  # the std of all rewards

    def push(self, x):
        self.n += 1  # update the number of reward signals collected
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n  # update mean
            self.s = self.s + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.s / (self.n - 1) if self.n > 1 else np.square(self.mean))  # update std


dqn = DQN()
running_stat = RunningStat()
device = torch.device('cuda')
dqn.eval_net.to(device)
dqn.target_net.to(device)

i_episode = 0

episode_return_total = np.zeros(0)  # episode return recorder

while True:
    state = env.reset()  # reset the environment
    last_done = np.array([[0., 0, 0]])
    episode_return = 0  # cumulative reward

    while True:
        action = np.zeros((1, num_agent))
        # choose actions
        for i in range(num_agent):
            temp = state[:, i:i + 1].reshape(1, -1)
            action_temp = dqn.choose_action(temp, max(epsilon_origin - epsilon_decrement * i_episode, 0.1),
                                            env.action_chosen[0, i])
            action[0, i] = action_temp

        agent_position = env.agent_position
        agent_orientation_angle = np.arctan2(env.agent_orientation[1:2, :], env.agent_orientation[0:1, :])
        target_position = env.target_position
        target_orientation_angle = np.arctan2(env.target_orientation[1:2, :], env.target_orientation[0:1, :])
        ######################
        # execute actions
        state_next, reward, done = env.step(action)
        # env.render()  # render the training process
        ######################
        agent_position_next = env.agent_position
        agent_orientation_angle_next = np.arctan2(env.agent_orientation[1:2, :], env.agent_orientation[0:1, :])
        target_position_next = env.target_position
        target_orientation_angle_next = np.arctan2(env.target_orientation[1:2, :], env.target_orientation[0:1, :])
        for i in range(num_agent):
            if not np.ravel(last_done)[i]:  # if the pursuer is active
                episode_return += np.ravel(reward)[i]
                if action[0, i] > 23:
                    # combine the task reward and the macro action bonus
                    reward[0, i] += option_bonus * ((action[0, i] - 24) % 20)
                running_stat.push(reward[:, i])
                reward[0, i] = np.clip(reward[0, i] / (running_stat.std + 1e-8), -10, 10)  # reward normalization

                temp1 = np.array(
                    [[agent_position[0, i] / 5000, agent_position[1, i] / 5000, agent_orientation_angle[0, i]]])
                temp2 = np.array([[agent_position_next[0, i] / 5000, agent_position_next[1, i] / 5000,
                                   agent_orientation_angle_next[0, i]]])
                temp3 = np.array(
                    [[target_position[0, 0] / 5000, target_position[1, 0] / 5000, target_orientation_angle[0, 0]]])
                temp4 = np.array([[target_position_next[0, 0] / 5000, target_position_next[1, 0] / 5000,
                                   target_orientation_angle_next[0, 0]]])
                transition = np.hstack((state[:, i].reshape((1, num_state)), action[:, i].reshape(1, 1),
                                        reward[:, i:i + 1], state_next[:, i].reshape((1, num_state)), done[:, i:i + 1],
                                        temp1, temp2, temp3, temp4))
                for j in np.delete(np.array(range(num_agent)), i):
                    temp1 = np.array(
                        [[agent_position[0, j] / 5000, agent_position[1, j] / 5000, agent_orientation_angle[0, j]]])
                    temp2 = np.array([[agent_position_next[0, j] / 5000, agent_position_next[1, j] / 5000,
                                       agent_orientation_angle_next[0, j]]])
                    temp3 = action[:, j].reshape((1, 1))
                    temp4 = state_next[:, j].reshape((1, num_state))
                    transition = np.hstack((transition, temp1, temp2, temp3, temp4))
                # Store the transitions into the reply memory. For simplicity, we only store tau transitions during the
                # execution of a tau-timestep macro action. But it is obvious we can store all macro actions that are
                # derived  from the same expert policy at each timestep, which means that totally tau*tau0 transitions
                # we acquire during the this tau timesteps.
                dqn.store_transition(transition)

        if np.all(done):
            if dqn.memory.sumtree.n_entries == memory_size:  # if the replay memory collects enough transitions
                for _ in range(1000):
                    dqn.learn(i_episode)  # train the network
                i_episode += 1
            break
        state = state_next
        last_done = done
    # if the replay memory doesn't collect enough transitions, print information
    if dqn.memory.sumtree.n_entries < memory_size:
        temp = "collecting experiences: " + str(dqn.memory.sumtree.n_entries) + ' / ' + str(memory_size)
        print(temp)
    # if the replay memory collects enough transitions, plot cumulative reward and print information
    if dqn.memory.sumtree.n_entries == memory_size:
        episode_return_total = np.hstack((episode_return_total, episode_return))
        #     plt.figure(1)
        #     plt.cla()
        #     plt.plot(episode_return_total)
        #     plt.show(block=False)
        #     plt.pause(0.01)
        print('i_episode: ', i_episode, 'episode_return: ', round(episode_return, 2))
    # periodically save networks
    if i_episode % 500 == 0:
        net = dqn.eval_net
        string = str(i_episode) + '.pt'
        torch.save(net.state_dict(), string)
        string = str(i_episode) + '.txt'
        np.savetxt(string, episode_return_total)
    # kill the training process, then calculate the success rate of all saved policies
    if i_episode == episode_max:
        validation('EASpace_DQN')
        break
