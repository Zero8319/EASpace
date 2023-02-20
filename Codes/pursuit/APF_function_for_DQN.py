import numpy as np


def attract(self_position, target_position):
    '''
    This function is used to calculate attractive force.
    Input:
        self_position: the position of pursuer
        target_position: the position of evader
    Output:
        F: the attractive force
    '''
    F = (target_position - self_position) / np.linalg.norm(target_position - self_position)
    return F


def repulse(self_position, obstacle_closest, influence_range, scale_repulse):
    '''
    This function is used to calculate repulsive force.
    Input:
        self_position: the position of pursuer
        obstacle_closest: the position of the nearest obstacle
        influence_range: the influence range of obstacles
        scale_repulse: the scale factor of repulsive force
    Output:
        F: the repulsive force
    '''
    F = np.matmul((1 / (np.linalg.norm(self_position - obstacle_closest) - 100) - 1 / influence_range) / (
            np.linalg.norm(self_position - obstacle_closest) - 100) ** 2 * (
                          self_position - obstacle_closest) / np.linalg.norm(self_position - obstacle_closest),
                  scale_repulse.transpose())
    if np.linalg.norm(self_position - obstacle_closest) < influence_range:
        # if the pursuer is within the obstacle's influence range
        return F
    else:
        return np.zeros((2, scale_repulse.shape[0]))


def individual(self_position, friend_position, individual_balance, r_perception):
    '''
    This function is used to calculate inter-individual force.
    Input:
        self_position: the position of pursuer
        friend_position:  the positions of teammates
        individual_balance: lambda
        r_perception: d_s
    Output:
        F: the individual force
    '''
    F = np.zeros((2, individual_balance.shape[0], 0))
    for i in range(friend_position.shape[1]):
        temp = np.matmul((friend_position[:, i:i + 1] - self_position) / np.linalg.norm(
            friend_position[:, i:i + 1] - self_position), (0.5 - individual_balance / (
                np.linalg.norm(friend_position[:, i:i + 1] - self_position) - 200)).transpose())
        if np.linalg.norm(friend_position[:, i:i + 1] - self_position) < r_perception:
            # mask distant teammates
            F = np.concatenate((F, temp.reshape((2, -1, 1))), axis=2)
    if F.size == 0:
        F = np.zeros((2, individual_balance.shape[0], 1))
    return np.mean(F, axis=2, keepdims=True).reshape((2, -1))


def generate_boundary(point1, point2, point3, point4, step):
    '''
    This function is used to generate obstacle points which make up a rectangular.
    Input:
        point1: the coordinate of the left bottom
        point2: the coordinate of the right bottom
        point3: the coordinate of the left top
        point4: the coordinate of the right top
        step: spacing between values
    Output:
        boundary: obstacle points
    '''
    temp1 = np.ravel(np.arange(point1[0], point2[0] + 1, step))
    temp2 = np.ravel(np.arange(point2[1], point3[1] + 1, step))
    boundary12 = np.vstack((temp1, np.ones_like(temp1) * point1[1]))
    boundary23 = np.vstack((np.ones_like(temp2) * point2[0], temp2))
    boundary34 = np.vstack((np.flipud(temp1), np.ones_like(temp1) * point3[1]))
    boundary41 = np.vstack((np.ones_like(temp2) * point4[0], np.flipud(temp2)))

    boundary = np.hstack((boundary12, boundary23, boundary34, boundary41))
    return boundary


def wall_follow(self_orientation, F_repulse, F_individual):
    '''
    Wall following rules for pursuers.
    Input:
        self_orientation: the pursuer's heading
        F_repulse: the repulsive force of the pursuer
        F_individual: the inter-individual force of the pursuer
    Output:
        rotate_vector: the resultant force according to wall following rules
    '''
    self_orientation = np.ravel(self_orientation)
    # calculate n_1 and n_2
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
    rotate_vector2 = -1 * rotate_vector1
    # choose between n_1 and n_2
    temp1 = np.linalg.norm(rotate_vector1 - self_orientation)
    temp2 = np.linalg.norm(rotate_vector2 - self_orientation)
    if np.linalg.norm(F_individual) < 1:  # if inter-individual force is less threshold B
        if temp1 > temp2:  # choose according to the heading
            return rotate_vector2
        else:
            return rotate_vector1
    else:  # if inter-individual force exceeds threshold B,choose according to the inter-individual force
        if np.dot(np.ravel(rotate_vector1), np.ravel(F_individual)) > 0:  #
            return rotate_vector1
        else:
            return rotate_vector2


def wall_follow_for_escaper(F_repulse, target_orientation, distance_from_nearest_agent, F_escape,
                            distance_from_nearest_obstacle):
    '''
    Wall following rules for the evader.
    Input:
        F_repulse: the repulsive force of the evader
        target_orientation: the evader's heading
        distance_from_nearest_agent: the distance between the evader and the nearest pursuer
        F_escape: the escape force or zigzagging force of the evader
        distance_from_nearest_obstacle: the distance between the evader and the nearest obstacle
    Output:
        final: the resultant force according to wall following rules
    '''
    # calculate n_1 and n_2
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
    rotate_vector2 = -1 * rotate_vector1
    # choose between n_1 and n_2
    if np.dot(np.ravel(target_orientation), np.ravel(rotate_vector1)) > 0:
        # if n_1 forms a smaller angle with the evader's heading
        final = rotate_vector1
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector2)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            # if any pursuer is close and it is in the front of evader, choose n_2
            final = rotate_vector2
    else:
        # if n_2 forms a smaller angle with the evader's heading
        final = rotate_vector2
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector1)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            # if any pursuer is close and it is in the front of evader, choose n_1
            final = rotate_vector1
    if distance_from_nearest_obstacle < 150:
        # if the evader is too close to obstacles, add another force to avoid collisions
        final = final + F_repulse
    return final


def APF_decision(self_position, friend_position, target_position, obstacle_closest, scale_repulse, individual_balance,
                 r_perception):
    '''
    This function is used to calculate the attractive force, the repulsive force, the inter-individual force.
    Input:
        self_position: the position of the pursuer
        friend_position: the positions of teammates
        target_position: the position of the evader
        obstacle_closest: the position of the nearest obstacle
        scale_repulse: the scale factor of repulsive force
        individual_balance: the parameter of inter-individual force, lambda
        r_perception: d_s
    Output:
        F_attract: the attractive forcce
        F_repulse: the repulsive force
        F_individual: the inter-individual force
        F: the resultant force of above three forces
    '''
    influence_range = 800  # the influence range of obstacles
    F_attract = attract(self_position, target_position)  # calculate the attractive force
    # calculate the repulsive force
    F_repulse = repulse(self_position, obstacle_closest, influence_range, scale_repulse)
    # calculate the inter-individual force
    F_individual = individual(self_position, friend_position, individual_balance, r_perception)
    # calculate the resultant force
    F = F_attract + F_repulse + F_individual
    return F_attract, F_repulse, F_individual, F


def total_decision(agent_position, agent_orientation, obstacle_closest, target_position, scale_repulse,
                   individual_balance, r_perception, which):
    '''
    This function is used to calculate resultant force, considering the wall following rules.
    Input:
        agent_position: the positions of pursuers
        agent_orientation: the headings of pursuers
        obstacle_closest: the positions of the closest obstacle for all pursuers
        target_position: the position of evader
        scale_repulse: the parameters(eta) for all pursuers
        individual_balance: the parameters (lambda) for all pursuers
        r_perception: d_s
    Output: the resultant force for all pursuers
    '''
    F = np.zeros((2, agent_position.shape[1], individual_balance.shape[0]))  # resultant force buffer
    wall_following = np.zeros((1, agent_position.shape[1], individual_balance.shape[
        0]))  # flag of whether pursuers move according to the wall following rules
    for i in np.ravel(which).tolist():
        self_position = agent_position[:, i:i + 1]
        friend_position = np.delete(agent_position, i, axis=1)
        self_orientation = agent_orientation[:, i:i + 1]
        # calculate APF forces
        F_attract, F_repulse, F_individual, F_total = APF_decision(self_position, friend_position, target_position,
                                                                   obstacle_closest[:, i:i + 1],
                                                                   scale_repulse[:, i:i + 1],
                                                                   individual_balance[:, i:i + 1], r_perception)
        for j in range(individual_balance.shape[0]):
            vector1 = F_attract[:, 0] + F_repulse[:, j]  # calculate F_ar
            vector2 = F_attract[:, 0]
            F_temp = F_total[:, j]
            # if np.dot(vector1, vector2) < 0:  # if the angle between F_ar and F_a exceeds 90 degree
            #     # move according to wall following rules
            #     F_temp = wall_follow(self_orientation, F_repulse[:, j], F_individual[:, j])
            #     wall_following[0, i, j] = True
            # else:
            #     F_temp = F_total[:, j]
            #     wall_following[0, i, j] = False
            F_temp = F_temp / np.linalg.norm(F_temp)  # normalize the resultant force
            F[:, i, j] = F_temp
    return F, wall_following
