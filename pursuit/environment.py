import numpy as np
import APF_function_for_DQN
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import escaper
import copy


class environment():
    def __init__(self, gamma, mode, algorithm):
        ####environment features#######
        self.mode = mode  # training mode or validation mode
        self.algorithm = algorithm
        self.num_agent = 3  # number of pursuers
        if self.algorithm in ['EASpace_DQN']:
            self.num_action = 24 + 20 + 20  # the number of actions in the integrated action space
        if self.algorithm in ['DQN', 'Shaping']:
            self.num_action = 24
        if self.algorithm in ['Caps']:
            self.num_action = 26
        self.num_state = 2 + 2 * 10 + (self.num_agent - 1) * 2 + 1  # dimension of state space
        self.t = 0  # timestep
        self.v = 300  # velocity of pursuers (mm/s)
        self.delta_t = 0.2  # time interval (s)
        self.gamma = gamma  # discount factor
        self.r_perception = 2000  # sense range of pursuers
        #######agent features#######
        if self.algorithm in ['EASpace_DQN', 'Caps']:
            self.action_chosen = np.ones((1, self.num_agent)) * -1
        self.agent_orientation = np.zeros((2, self.num_agent))
        self.agent_orientation_last = np.zeros((2, self.num_agent))
        self.agent_orientation_origin = np.zeros((2, self.num_agent))
        self.agent_position = np.zeros((2, self.num_agent))
        self.agent_position_last = np.zeros((2, self.num_agent))
        self.agent_position_origin = np.zeros((2, self.num_agent))
        self.obstacle_closest = np.zeros((2, self.num_agent))
        self.distance_from_target = np.zeros((1, self.num_agent))
        self.distance_from_target_last = np.zeros((1, self.num_agent))
        self.done = np.zeros((1, self.num_agent))
        self.state = np.zeros((self.num_state, self.num_agent))

        #######evader features########
        self.target_position = np.zeros((2, 1))
        self.target_orientation = np.zeros((2, 1))
        self.escaper_slip_flag = 0
        self.escaper_wall_following = 0
        self.escaper_zigzag_flag = 0
        self.last_e = np.zeros((2, 1))
        self.zigzag_count = 0
        self.zigzag_last = np.zeros((2, 1))
        #######obstacle features#######
        self.boundary = APF_function_for_DQN.generate_boundary(np.array([[0.0], [0]]), np.array([[3600], [0]]),
                                                               np.array([[3600], [5000]]), np.array([[0], [5000]]), 51)
        theta = list(range(25, 90))
        x = 3200 * np.cos(np.radians(theta))
        y = 1500 * np.sin(np.radians(theta))
        self.obstacle1 = np.vstack((x, y))
        self.obstacle2 = APF_function_for_DQN.generate_boundary(np.array([[3400.0], [1000]]),
                                                                np.array([[3600], [1000]]),
                                                                np.array([[3600], [1100]]), np.array([[3400], [1100]]),
                                                                11)
        self.obstacle3 = APF_function_for_DQN.generate_boundary(np.array([[1400.0], [2450]]),
                                                                np.array([[2200], [2450]]),
                                                                np.array([[2200], [2550]]), np.array([[1400], [2550]]),
                                                                11)
        self.obstacle4 = APF_function_for_DQN.generate_boundary(np.array([[900.0], [3900]]),
                                                                np.array([[1550], [3900]]),
                                                                np.array([[1550], [4000]]), np.array([[900], [4000]]),
                                                                11)
        self.obstacle5 = APF_function_for_DQN.generate_boundary(np.array([[2050.0], [3900]]),
                                                                np.array([[2700], [3900]]),
                                                                np.array([[2700], [4000]]), np.array([[2050], [4000]]),
                                                                11)

        self.obstacle_total = np.hstack(
            (self.boundary, self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle4, self.obstacle5))

    def reset(self):
        self.t = 0
        #######agent features#######
        self.action_chosen = np.ones((1, self.num_agent)) * -1
        self.agent_orientation = np.vstack((np.zeros((1, self.num_agent)), np.ones((1, self.num_agent))))
        self.agent_orientation_last = np.zeros((2, self.num_agent))
        self.agent_orientation_origin = self.agent_orientation  # original headings
        self.agent_position = np.array([[500., 900, 1300], [1200, 1150, 1100]])
        self.agent_position_last = np.zeros((2, self.num_agent))
        self.agent_position_origin = self.agent_position  # original positions
        self.obstacle_closest = np.zeros((2, self.num_agent))
        self.obstacle_closest10 = np.zeros((20, self.num_agent))
        self.distance_from_target = np.zeros((1, self.num_agent))
        self.distance_from_target_last = np.zeros((1, self.num_agent))
        self.done = np.zeros((1, self.num_agent))
        self.state = np.zeros((self.num_state, self.num_agent))

        #######evader features########
        self.target_position = np.zeros((2, 1))
        self.target_orientation = np.zeros((2, 1))
        self.escaper_slip_flag = 0
        self.escaper_wall_following = 0
        self.escaper_zigzag_flag = 0
        self.last_e = np.zeros((2, 1))
        self.zigzag_count = 0
        self.zigzag_last = np.zeros((2, 1))

        # initialize evader's positions and headings
        self.target_position = np.random.random((2, 1))
        self.target_position[0] = self.target_position[0] * 3200 + 200
        self.target_position[1] = self.target_position[1] * 600 + 4200
        self.target_orientation = np.array([[0.], [1]])

        self.update_feature()
        self.update_state()  # update environment's state

        return self.state

    def reward(self):
        reward = np.zeros((1, self.num_agent))  # reward buffer
        done = np.zeros((1, self.num_agent))  # done buffer
        position_buffer = copy.deepcopy(self.agent_position)

        success_flag = np.any(self.done)

        for i in range(self.num_agent):
            reward2 = 0  # r_time
            reward3 = 0  # r_tm
            reward4 = 0  # r_o
            reward5 = 0  # r_pot
            if success_flag:
                success_range = 300
            else:
                success_range = 200

            potential = 0
            potential_last = 0
            done_temp = 0
            if self.distance_from_target[0, i] < success_range:  # if the distance from the evader is less than d_c
                potential = 50
                done_temp = 1.  # the pursuer captures the evader successfully
            if self.distance_from_target_last[0, i] < success_range:
                potential_last = 50
            reward1 = potential - potential_last  # r_main
            ##############################################################
            if not done_temp:
                if self.t == 1000: # the maximal timestep
                    done_temp = 2.  # timeout
                ##############################################################
                if np.arccos(np.clip(np.dot(np.ravel(self.agent_orientation_last[:, i:i + 1]),
                                            np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                    self.agent_orientation_last[:, i:i + 1]) / np.linalg.norm(self.agent_orientation[:, i:i + 1]),
                                     -1, 1)) > np.radians(45):  # if the pursuer's steering angle exceeds 45
                    reward2 = -5
                else:
                    reward2 = 0
                ##############################################################
                if np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) > 150:
                    # if the distance from the nearest obstacle exceeds 150 mm
                    reward3 = 0
                elif np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
                    # if the distance from the nearest obstacle is less than 100 mm
                    reward3 = -50
                    if self.mode == 'Train':
                        # the pursuer collides and be moved to its original position
                        position_buffer[:, i:i + 1] = self.agent_position_origin[:, i:i + 1]
                    if self.mode == 'Valid':
                        # the pursuer is inactive
                        done_temp = 3.
                else:
                    reward3 = -2
                ##############################################################
                if np.amin(np.linalg.norm(self.agent_position[:, i:i + 1] - np.delete(self.agent_position, i, axis=1),
                                          axis=0)) > 200:
                    # if the distance from the nearest teammate exceeds 200 mm
                    reward4 = 0
                else:
                    if self.action_chosen[0, i] >= 44 and self.agent_position[1, i] < 1500:
                        reward4 = 0
                    else:
                        reward4 = -50
                    if self.mode == 'Train':
                        # the pursuer collides and be moved to its original position
                        position_buffer[:, i:i + 1] = self.agent_position_origin[:, i:i + 1]
                    if self.mode == 'Valid':
                        # the pursuer is inactive
                        done_temp = 3.
                ##############################################################
                if self.distance_from_target[0, i] < 2500:
                    reward5 = (self.distance_from_target_last[0, i] - self.distance_from_target[0, i]) / 200
                ##############################################################

            reward[0, i] = reward1 + reward2 + reward3 + reward4 + reward5  # the total reward
            done[0, i] = done_temp

        # in the training mode, initialize the collided pursuer, in validation mode, do nothing
        self.agent_position = position_buffer
        self.done = done

        return reward, done

    def step(self, action):
        self.action_chosen = copy.deepcopy(action)
        self.t += 1
        self.update_feature_last()

        ######agent#########
        if self.algorithm in ['EASpace_DQN']:
            F = np.zeros((2, self.num_agent))
            for i in range(self.num_agent):
                if action[0, i] < 24:
                    # primitive actions
                    agent_orientation_angle = np.arctan2(self.agent_orientation[1, i], self.agent_orientation[0, i])
                    F[0, i] = np.cos(np.radians(action[0, i] * 360 / 24) + agent_orientation_angle)
                    F[1, i] = np.sin(np.radians(action[0, i] * 360 / 24) + agent_orientation_angle)
                elif action[0, i] < 24 + 20:
                    # APF
                    F_APF = self.from_action_to_APF()
                    F_APF = F_APF[:, i, :]
                    agent_orientation = self.agent_orientation[:, i:i + 1]
                    temp = np.radians(40)
                    if np.arccos(np.clip(np.dot(np.ravel(agent_orientation), np.ravel(F_APF)) / np.linalg.norm(
                            agent_orientation) / np.linalg.norm(F_APF), -1, 1)) > temp:
                        rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
                        temp1 = np.matmul(rotate_matrix, agent_orientation)
                        rotate_matrix = np.array(
                            [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
                        temp2 = np.matmul(rotate_matrix, agent_orientation)
                        if np.dot(np.ravel(temp1), np.ravel(F_APF)) > np.dot(np.ravel(temp2), np.ravel(F_APF)):
                            F_APF = temp1
                        else:
                            F_APF = temp2
                    F[:, i:i + 1] = F_APF
                else:
                    # wall following
                    agent_orientation = self.agent_orientation[:, i:i + 1]
                    F_repulse = self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]
                    F_repulse = F_repulse / np.linalg.norm(np.ravel(F_repulse))
                    rotate_matrix = np.array([[0, -1], [1, 0]])
                    F_wall_following = np.matmul(rotate_matrix, F_repulse)
                    # rotate_vector2 = -1 * rotate_vector1
                    # if np.dot(np.ravel(agent_orientation), np.ravel(rotate_vector1)) > 0:
                    #     F_wall_following = rotate_vector1
                    # else:
                    #     F_wall_following = rotate_vector2
                    if np.linalg.norm(self.agent_position[:, i] - self.obstacle_closest[:, i]) < 200:
                        F_wall_following = F_wall_following + F_repulse
                    if np.linalg.norm(self.agent_position[:, i] - self.obstacle_closest[:, i]) > 300:
                        F_wall_following = F_wall_following - F_repulse
                    temp = np.radians(40)
                    if np.arccos(
                            np.clip(np.dot(np.ravel(agent_orientation), np.ravel(F_wall_following)) / np.linalg.norm(
                                agent_orientation) / np.linalg.norm(F_wall_following), -1, 1)) > temp:

                        rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
                        temp1 = np.matmul(rotate_matrix, agent_orientation)
                        rotate_matrix = np.array(
                            [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
                        temp2 = np.matmul(rotate_matrix, agent_orientation)
                        if np.dot(np.ravel(temp1), np.ravel(F_wall_following)) > np.dot(np.ravel(temp2),
                                                                                        np.ravel(F_wall_following)):
                            F_wall_following = temp1
                        else:
                            F_wall_following = temp2
                    F[:, i:i + 1] = F_wall_following
        if self.algorithm in ['Caps']:
            F = np.zeros((2, self.num_agent))
            for i in range(self.num_agent):
                if action[0, i] < 24:
                    agent_orientation_angle = np.arctan2(self.agent_orientation[1, i], self.agent_orientation[0, i])
                    F[0, i] = np.cos(np.radians(action[0, i] * 360 / 24) + agent_orientation_angle)
                    F[1, i] = np.sin(np.radians(action[0, i] * 360 / 24) + agent_orientation_angle)
                elif action[0, i] == 24:
                    F_APF = self.from_action_to_APF()
                    F_APF = F_APF[:, i, :]
                    agent_orientation = self.agent_orientation[:, i:i + 1]
                    temp = np.radians(40)
                    if np.arccos(np.clip(np.dot(np.ravel(agent_orientation), np.ravel(F_APF)) / np.linalg.norm(
                            agent_orientation) / np.linalg.norm(F_APF), -1, 1)) > temp:
                        rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
                        temp1 = np.matmul(rotate_matrix, agent_orientation)
                        rotate_matrix = np.array(
                            [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
                        temp2 = np.matmul(rotate_matrix, agent_orientation)
                        if np.dot(np.ravel(temp1), np.ravel(F_APF)) > np.dot(np.ravel(temp2), np.ravel(F_APF)):
                            F_APF = temp1
                        else:
                            F_APF = temp2
                    F[:, i:i + 1] = F_APF
                else:
                    agent_orientation = self.agent_orientation[:, i:i + 1]
                    F_repulse = self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]
                    F_repulse = F_repulse / np.linalg.norm(np.ravel(F_repulse))
                    rotate_matrix = np.array([[0, -1], [1, 0]])
                    F_wall_following = np.matmul(rotate_matrix, F_repulse)
                    if np.linalg.norm(self.agent_position[:, i] - self.obstacle_closest[:, i]) < 200:
                        F_wall_following = F_wall_following + F_repulse
                    if np.linalg.norm(self.agent_position[:, i] - self.obstacle_closest[:, i]) > 300:
                        F_wall_following = F_wall_following - F_repulse
                    temp = np.radians(40)
                    if np.arccos(
                            np.clip(np.dot(np.ravel(agent_orientation), np.ravel(F_wall_following)) / np.linalg.norm(
                                agent_orientation) / np.linalg.norm(F_wall_following), -1, 1)) > temp:
                        rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
                        temp1 = np.matmul(rotate_matrix, agent_orientation)
                        rotate_matrix = np.array(
                            [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
                        temp2 = np.matmul(rotate_matrix, agent_orientation)
                        if np.dot(np.ravel(temp1), np.ravel(F_wall_following)) > np.dot(np.ravel(temp2),
                                                                                        np.ravel(F_wall_following)):
                            F_wall_following = temp1
                        else:
                            F_wall_following = temp2
                    F[:, i:i + 1] = F_wall_following
        if self.algorithm in ['DQN', 'Shaping']:
            agent_orientation_angle = np.arctan2(self.agent_orientation[1, :], self.agent_orientation[0, :]).reshape(1,
                                                                                                                     self.num_agent)
            temp1 = np.cos(np.radians(action * 360 / 24) + agent_orientation_angle)
            temp2 = np.sin(np.radians(action * 360 / 24) + agent_orientation_angle)
            F = np.vstack((temp1, temp2))
        agent_position_buffer = np.zeros((2, self.num_agent))
        for i in range(self.num_agent):
            if self.done[0, i]:  # if the pursuer has been captured, it will not move
                pass
            else:
                agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t

        #######escaper########
        # calculate the evader's displacement according to the escaping policy
        F_escaper, zigzag_count, zigzag_last, escaper_zigzag_flag, escaper_wall_following, escaper_slip_flag, distance_from_nearest_obstacle, last_e = escaper.escaper(
            self.agent_position, self.target_position,
            self.target_orientation, self.obstacle_total,
            self.num_agent, self.zigzag_count, self.zigzag_last, self.last_e, self.escaper_slip_flag)
        self.zigzag_last = zigzag_last
        self.zigzag_count = zigzag_count
        self.escaper_zigzag_flag = escaper_zigzag_flag
        self.escaper_wall_following = escaper_wall_following
        self.escaper_slip_flag = escaper_slip_flag
        self.last_e = last_e
        #####update#####
        self.agent_position = self.agent_position + agent_position_buffer  # update pursuers'positions
        self.agent_orientation = F  # update pursuers' headings

        if np.any(self.done) or distance_from_nearest_obstacle < 30:
            # if the evader is captured or collides with obstacles
            pass
        else:
            self.target_position = self.target_position + F_escaper * self.delta_t  # update the evader's position
            self.target_orientation = F_escaper  # update the evader's heading
        #####################################################################################
        self.update_feature()
        reward, _ = self.reward()  # calculate reward function
        self.update_feature()
        self.update_state()  # update environment's state
        return self.state, reward, self.done

    def render(self):
        plt.figure(1)
        plt.cla()
        ax = plt.gca()
        plt.xlim([-100, 3700])
        plt.ylim([-100, 5100])
        ax.set_aspect(1)
        # plot obstacles and boundary
        plt.plot(self.obstacle1[0, :], self.obstacle1[1, :], 'black')
        plt.plot(self.obstacle2[0, :], self.obstacle2[1, :], 'black')
        plt.plot(self.obstacle3[0, :], self.obstacle3[1, :], 'black')
        plt.plot(self.obstacle4[0, :], self.obstacle4[1, :], 'black')
        plt.plot(self.obstacle5[0, :], self.obstacle5[1, :], 'black')
        plt.plot(self.boundary[0, :], self.boundary[1, :], 'black')
        # plot evader
        circle = mpatches.Circle(np.ravel(self.target_position), 100)
        ax.add_patch(circle)
        # plot pursuers
        color = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(self.num_agent):
            circle = mpatches.Circle(self.agent_position[:, i], 100, facecolor=color[i])
            ax.add_patch(circle)
            if not self.done[0, i]:
                plt.quiver(self.agent_position[0, i], self.agent_position[1, i], self.agent_orientation[0, i],
                           self.agent_orientation[1, i], scale=10)
        plt.show(block=False)
        # plt.savefig(str(self.t))#whether save figures
        plt.pause(0.001)

    def update_state(self):
        self.state = np.zeros((self.num_state, self.num_agent))  # clear the environment state
        for i in range(self.num_agent):
            # the distance form the nearest obstacle
            state = np.zeros((self.num_state,))  # state buffer
            for j in range(10):
                temp1 = self.obstacle_closest10[j * 2:j * 2 + 2, i:i + 1] - self.agent_position[:, i:i + 1]
                # the bearing of the nearest obstacle
                angle1 = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp1), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                            temp1) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
                if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp1)) > 0:
                    pass
                else:
                    angle1 = -angle1
                state[j * 2] = np.linalg.norm(temp1) / 5000
                state[j * 2 + 1] = angle1
                # the distance from evader
            temp2 = self.target_position - self.agent_position[:, i:i + 1]
            # the bearing of evader
            angle2 = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp2), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp2) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp2)) > 0:
                pass
            else:
                angle2 = -angle2

            state[20] = np.linalg.norm(temp2) / 5000  # update state
            state[21] = angle2

            friends_position = np.delete(self.agent_position, i, axis=1)  # teammate positions
            for j in range(self.num_agent - 1):
                friend_position = friends_position[:, j:j + 1]  # teammate position
                self_position = self.agent_position[:, i:i + 1]
                self_orientation = self.agent_orientation[:, i:i + 1]
                temp = friend_position - self_position
                distance = np.linalg.norm(temp)  # the distance from teammate
                # the bearing of teammate
                angle = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp), np.ravel(self_orientation)) / distance / np.linalg.norm(
                            self_orientation), -1, 1)) / np.pi
                if np.cross(np.ravel(self_orientation), np.ravel(temp)) > 0:
                    pass
                else:
                    angle = -angle
                # mask distant teammates and update state
                if distance < self.r_perception:
                    state[23 + 2 * j] = np.linalg.norm(temp) / 5000
                    state[24 + 2 * j] = np.array(angle)
                else:
                    state[23 + 2 * j] = 2
                    state[24 + 2 * j] = 0
            if np.any(self.done == 1):
                state[22] = 1
            else:
                state[22] = 0
            self.state[:, i] = state

    def update_feature(self):
        obstacle_closest = np.zeros((2, self.num_agent))
        obstacle_closest10 = np.zeros((20, self.num_agent))
        for i in range(self.num_agent):
            # the index of nearest obstacle
            temp = np.argmin(np.linalg.norm(self.obstacle_total - self.agent_position[:, i:i + 1], axis=0))
            # the position of the nearest obstacle
            obstacle_closest[:, i:i + 1] = self.obstacle_total[:, temp:temp + 1]
            obstacle_total = self.obstacle_total
            for j in range(10):
                temp = np.argmin(np.linalg.norm(obstacle_total - self.agent_position[:, i:i + 1], axis=0))
                # the position of the nearest obstacle
                obstacle_closest10[j * 2:j * 2 + 2, i:i + 1] = obstacle_total[:, temp:temp + 1]
                obstacle_total = np.delete(obstacle_total, temp, axis=1)
        self.obstacle_closest = obstacle_closest
        self.obstacle_closest10 = obstacle_closest10
        self.distance_from_target = np.linalg.norm(self.agent_position - self.target_position, axis=0,
                                                   keepdims=True)  # the distance between the evader and pursuers

    def update_feature_last(self):
        self.agent_position_last = copy.deepcopy(self.agent_position)
        self.agent_orientation_last = copy.deepcopy(self.agent_orientation)
        self.distance_from_target_last = copy.deepcopy(self.distance_from_target)

    def from_action_to_APF(self, which=np.array([[0, 1, 2]])):
        scale_repulse = np.ones((1, self.num_agent)) * 1e7  # eta buffer
        individual_balance = np.ones((1, self.num_agent)) * 1200  # lambda buffer
        # calculate APF resultant forces
        F, wall_following = APF_function_for_DQN.total_decision(self.agent_position,
                                                                self.agent_orientation,
                                                                self.obstacle_closest,
                                                                self.target_position,
                                                                scale_repulse,
                                                                individual_balance,
                                                                self.r_perception, which)

        return F
