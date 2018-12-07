import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(__file__)
rewards_list = []


def getInputs():
    filename = "pipe_world.txt"
    path = ROOT + "/" + filename
    if not os.path.exists(path):
        print("Please Enter Correct File name !")
        return
    maze = []
    count = 0
    start_position = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            # maze.append(list(map(str, line.split(" "))))
            maze.append([])
            temp = 0
            for s in line:
                maze[count].append(s)
                if s == 'S':
                    start_position.append(count)
                    start_position.append(temp)
                temp += 1
            count += 1
    feature_based_Q_Learning(maze, start_position, 0.9, 0.9, 0.9)


def feature_based_Q_Learning(maze, start_position, learning_rate, policy_randomness, future_discount):
    feature_table = np.zeros([len(maze), len(maze[0]), 4, 2])
    initFeatureTable(feature_table, maze)
    weight = [0, 0]
    for episode in range(1, 10001):
        life = True
        cur_position = [start_position[0], start_position[1]]
        if episode % 1000 == 0:
            learning_rate = updateLearnRate(learning_rate, episode)
        if episode % 200 == 0:
            policy_randomness = updatePolicyRandomness(policy_randomness, episode)
        if episode % 100 == 0:
            # evaluate the policy
            evaluate(maze, start_position, feature_table, weight)
        steps = 0
        while life:
            action = 0
            if np.random.uniform() < policy_randomness:
                action = randomAction(cur_position, feature_table)
            else:
                action = predictAction(cur_position, feature_table, weight, maze)
            new_position = getNewPosition(cur_position, action, feature_table)
            reward = getReward(new_position, maze)
            updateWeight(cur_position, new_position, action, feature_table, weight, reward, learning_rate,
                         future_discount)
            cur_position = new_position
            steps += 1
            life = isContinue(new_position, maze, steps)


def evaluate(maze, start_position, feature_table, weight):
    pass


def updateWeight(cur_position, new_position, action, feature_table, weight, reward, learning_rate, future_discount):
    cur_feature_vector = feature_table[cur_position[0]][cur_position[1]][action]
    cur_qvalue = cur_feature_vector[0] * weight[0] + cur_feature_vector[1] * weight[1]
    # calculate the max qvalue of new position
    max_qvalue = getMaxQValue(new_position, feature_table, weight)
    val = (reward + future_discount * max_qvalue - cur_qvalue) * learning_rate
    cur_feature_vector[0] *= val
    cur_feature_vector[1] *= val
    weight[0] += cur_feature_vector[0]
    weight[1] += cur_feature_vector[1]


def getMaxQValue(new_position, feature_table, weight):
    qvalue_list = feature_table[new_position[0]][new_position[1]]
    dict = {}
    for i in range(len(qvalue_list)):
        if isNan(qvalue_list[i][0]):
            continue
        dict[i] = qvalue_list[i]
    max_index = 0
    max_qvalue = -sys.maxsize - 1
    for key in dict:
        vector = dict[key]
        q_value = vector[0] * weight[0] + vector[1] * weight[0]
        if q_value > max_qvalue:
            max_qvalue = q_value
            max_action = int(key)
    return max_qvalue


def getNewPosition(cur_position, action, feature_table):  # probally have issues
    # Sliper
    new_position = [0, 0]
    new_position[0] = cur_position[0]
    new_position[1] = cur_position[1]
    if action == 0:
        new_position[0] = cur_position[0] - 1
        if predictSlip():
            # Go Down and Up , need to consider left and right border
            if isYBounded(new_position[1], feature_table):
                new_position[1] = cur_position[1] + 1 if cur_position[1] == 0 else cur_position[1] - 1
            else:
                if random.randint(0, 1) == 1:
                    new_position[1] = cur_position[1] + 1
                else:
                    new_position[1] = cur_position[1] - 1
    elif action == 1:
        new_position[0] = cur_position[0] + 1
        if predictSlip():
            # Go Down and Up , need to consider left and right border
            if isYBounded(new_position[1], feature_table):
                new_position[1] = cur_position[1] + 1 if cur_position[1] == 0 else cur_position[1] - 1
            else:
                if random.randint(0, 1) == 1:
                    new_position[1] = cur_position[1] + 1
                else:
                    new_position[1] = cur_position[1] - 1
    elif action == 2:
        # Go left and right, need to consider up and bottom border
        new_position[1] = cur_position[1] - 1
        if predictSlip():
            if isXBounded(new_position[0], feature_table):
                new_position[0] = cur_position[0] + 1 if cur_position[0] == 0 else cur_position[0] - 1
            else:
                if random.randint(0, 1) == 1:
                    new_position[0] = cur_position[0] + 1
                else:
                    new_position[0] = cur_position[0] - 1
    elif action == 3:
        new_position[1] = cur_position[1] + 1
        if predictSlip():
            if isXBounded(new_position[0], feature_table):
                new_position[0] = cur_position[0] + 1 if cur_position[0] == 0 else cur_position[0] - 1
            else:
                if random.randint(0, 1) == 1:
                    new_position[0] = cur_position[0] + 1
                else:
                    new_position[0] = cur_position[0] - 1
    return new_position


def randomAction(cur_position, feature_table):
    x = cur_position[0]
    y = cur_position[1]
    list = []
    for z in range(len(feature_table[x][y])):
        if isNan(feature_table[x][y][z][0]):
            continue
        list.append(z)
    result = random.randint(0, len(list) - 1)
    return list[result]


def predictAction(cur_position, feature_table, weight, maze):
    x = cur_position[0]
    y = cur_position[1]
    dict = {}
    for z in range(len(feature_table[x][y])):
        if isNan(feature_table[x][y][z][0]):
            continue
        dict[z] = feature_table[x][y][z]
    print(dict)
    max_action = 0
    max_qvalue = -sys.maxsize - 1
    for key in dict:
        vector = dict[key]
        q_value = vector[0] * weight[0] + vector[1] * weight[0]
        if q_value > max_qvalue:
            max_qvalue = q_value
            max_action = int(key)
    return max_action


def initFeatureTable(feature_table, maze):
    # init the border
    for x in range(len(feature_table)):
        for y in range(len(feature_table[0])):
            if x == 0:  # in the top
                feature_table[x][y][0] = None
            if x == (len(feature_table) - 1):  # in the bottom
                feature_table[x][y][1] = None
            if y == 0:
                feature_table[x][y][2] = None
            if y == (len(feature_table[0]) - 1):
                feature_table[x][y][3] = None
    # init the feature vector
    md_plus = getMDPlus(maze)
    for x in range(len(feature_table)):  # row
        for y in range(len(feature_table[0])):  # column
            actions = []
            # state in the Mine
            if y == 0 or y == (len(feature_table[0]) - 1):
                continue
            for z in range(len(feature_table[0][0])):  # four actions
                # invalid action
                if isNan(feature_table[x][y][z][0]):
                    continue
                # Append its index(action) rather than value
                actions.append(z)
            for action in actions:
                position = [x, y]
                feature_vector = calculateFeatureVector(position, action, maze, md_plus)
                feature_table[x][y][action][0] = feature_vector[0]
                feature_table[x][y][action][1] = feature_vector[1]


def calculateFeatureVector(position, action, maze, md_plus):
    new_position = [position[0], position[1]]
    goal_position = getGoalPosition(maze)
    move(new_position, action)
    md_moved = calculateManhattanDistance(new_position, goal_position)
    f1 = md_moved / md_plus
    # calculate f2
    f2 = 0
    left_inverse_distance = 1 / position[1]
    right_inverse_distance = 1 / (len(maze[0]) - 1 - position[1])
    if action <= 1:
        f2 = min(left_inverse_distance, right_inverse_distance)
    elif action == 2:
        f2 = left_inverse_distance
    elif action == 3:
        f2 = right_inverse_distance
    return [f1, f2]


def move(new_position, action):
    if action == 0:
        new_position[0] = new_position[0] - 1
    elif action == 1:
        new_position[0] = new_position[0] + 1
    elif action == 2:
        new_position[1] = new_position[1] - 1
    elif action == 3:
        new_position[1] = new_position[1] + 1


def getMDPlus(maze):
    goal_position = getGoalPosition(maze)
    left_up = [0, 0]
    right_up = [0, len(maze[0]) - 1]
    left_down = [len(maze) - 1, 0]
    right_down = [len(maze) - 1, len(maze[0]) - 1]
    list = []
    list.append(calculateManhattanDistance(left_up, goal_position))
    list.append(calculateManhattanDistance(left_down, goal_position))
    list.append(calculateManhattanDistance(right_down, goal_position))
    list.append(calculateManhattanDistance(right_up, goal_position))
    return max(list)


def predictSlip():
    slip = random.randint(0, 9)
    if slip < 8:
        return False
    else:
        return True


def getGoalPosition(maze):
    position = [0, 0]
    for x in range(len(maze)):
        for y in range(len(maze[x])):
            if maze[x][y] == "G":
                position[0] = x
                position[1] = y
    return position


def calculateManhattanDistance(position1, position2):
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])


def isNan(val):
    return val is None or math.isnan(val)


def updatePolicyRandomness(policy_randomness, episode):
    return policy_randomness / (episode / 200 + 1)


def updateLearnRate(learning_rate, episode):
    return learning_rate / ((episode / 1000) + 1)


# left right border
def isYBounded(position, feature_table):
    if (position == 0) or (position == (len(feature_table[0]) - 1)):
        return True
    else:
        return False


# up bottom border
def isXBounded(position, feature_table):
    if (position == 0) or (position == (len(feature_table) - 1)):
        return True
    else:
        return False


def getReward(cur_position, maze):
    if maze[cur_position[0]][cur_position[1]] == "M":
        return -100
    elif maze[cur_position[0]][cur_position[1]] == "G":
        return 0
    else:
        return -1


def isContinue(cur_position, maze, steps):
    # Three Conditions
    # into Mine
    # is Alive?
    if maze[cur_position[0]][cur_position[1]] == "M":
        return False
    # is reach the goal?
    elif maze[cur_position[0]][cur_position[1]] == "G":
        # print("Reach The Goal")
        return False
    # is too mant steps?
    elif steps >= (len(maze) * len(maze[0])):
        return False
    else:
        return True


if __name__ == "__main__":
    getInputs()
