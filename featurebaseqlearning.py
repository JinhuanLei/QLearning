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
    Q_Learning(maze, start_position, 0.9, 0.9, 0.9)


def Q_Learning(maze, start_position, learning_rate, policy_randomness, future_discount):
    # 0 up 1 down 2 left 3 right
    # I did not update q_table in this algorithm, I just use it to judge explosive area
    q_table = np.zeros([len(maze), len(maze[0]), 4])
    initTable(q_table)
    weight = [0, 0]
    for episode in range(1, 10001):
        life = True
        cur_position = []
        cur_position.append(start_position[0])
        cur_position.append(start_position[1])
        if episode % 1000 == 0:
            learning_rate = updateLearnRate(learning_rate, episode)
        if episode % 200 == 0:
            policy_randomness = updatePolicyRandomness(policy_randomness, episode)
        if episode % 50 == 0:
            evaluateQTable(q_table, start_position, maze)
        steps = 0
        while life:
            action = 0
            if np.random.uniform() < policy_randomness:
                action = randomAction(cur_position, q_table)
            else:
                action = predictAction(cur_position, q_table, weight, maze)
            # Observe next state
            new_position = getNewPosition(cur_position, action, q_table)
            reward = getReward(new_position, maze)
            # print(cur_position)
            updateWeight(cur_position, new_position, action, q_table, learning_rate, future_discount, reward, weight,
                         maze)
            cur_position = new_position
            steps += 1
            life = isContinue(new_position, maze, steps)


def updateWeight(cur_position, new_position, action, q_table, learning_rate, future_discount, reward, weight, maze):
    feature_vector = getFeatureVector(cur_position, action, maze)
    q_value = feature_vector[0] * weight[0] + feature_vector[1] * weight[1]



def getF1(position, maze, action):
    md_plus = getMDPlus(maze)
    md_val = calculateManhattanDistance(position, getGoalPosition(maze))
    return md_val / md_plus


def move(new_position, action):
    if action == 0:
        new_position[0] = new_position[0] - 1
    elif action == 1:
        new_position[0] = new_position[0] + 1
    elif action == 2:
        new_position[1] = new_position[1] - 1
    elif action == 3:
        new_position[1] = new_position[1] + 1


def getF2(position, action):
    left_inverse_distance = 1 / position[0]
    right_inverse_distance = 1 / position[1]
    if action <= 1:
        return min(left_inverse_distance, right_inverse_distance)
    elif action == 2:
        return left_inverse_distance
    elif action == 3:
        return right_inverse_distance


def getFeatureVector(position, action, maze):
    feature_vector = []
    new_position = [position[0], position[1]]
    move(new_position, action)
    feature_vector.append(getF1(new_position, maze, action))
    feature_vector.append(getF2(position, action))
    return feature_vector


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


def calculateManhattanDistance(position1, position2):
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])


def getGoalPosition(maze):
    position = [0, 0]
    for x in range(len(maze)):
        for y in range(len(maze[x])):
            if maze[x][y] == "G":
                position[0] = x
                position[1] = y
    return position


def randomAction(cur_position, q_table):
    x = cur_position[0]
    y = cur_position[1]
    actions = q_table[x][y]
    list = []
    for i in range(0, len(actions)):
        a = actions[i]
        if a is None or math.isnan(a):
            continue
        list.append(i)
    result = random.randint(0, len(list) - 1)
    return list[result]


def evaluateQTable(q_table, start_position, maze):
    total_rewards = 0
    for episode in range(50):
        life = True
        cur_position = []
        cur_position.append(start_position[0])
        cur_position.append(start_position[1])
        steps = 0
        rewards = 0
        while life:
            action = predictAction(cur_position, q_table)
            # Observe next state
            new_position = getNewPosition(cur_position, action, q_table)
            reward = getReward(new_position, maze)
            rewards += reward
            cur_position = new_position
            steps += 1
            life = isContinue(new_position, maze, steps)
        total_rewards += rewards
    average_rewards = total_rewards / 50
    rewards_list.append(average_rewards)
    print(average_rewards)


def findMax(actions):
    maxVal = -sys.maxsize - 1
    for a in actions:
        if a is None:
            continue
        if a > maxVal:
            maxVal = a
    return maxVal


def updatePolicyRandomness(policy_randomness, episode):
    return policy_randomness / (episode / 200 + 1)


def updateLearnRate(learning_rate, episode):
    return learning_rate / ((episode / 1000) + 1)


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


def getReward(cur_position, maze):
    if maze[cur_position[0]][cur_position[1]] == "M":
        return -100
    elif maze[cur_position[0]][cur_position[1]] == "G":
        return 0
    else:
        return -1


def getNewPosition(cur_position, action, q_table):  # probally have issues
    # Sliper
    new_position = [0, 0]
    new_position[0] = cur_position[0]
    new_position[1] = cur_position[1]
    if action == 0:
        new_position[0] = cur_position[0] - 1
        if predictSlip():
            # Go Down and Up , need to consider left and right border
            if isYBounded(new_position[1], q_table):
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
            if isYBounded(new_position[1], q_table):
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
            if isXBounded(new_position[0], q_table):
                new_position[0] = cur_position[0] + 1 if cur_position[0] == 0 else cur_position[0] - 1
            else:
                if random.randint(0, 1) == 1:
                    new_position[0] = cur_position[0] + 1
                else:
                    new_position[0] = cur_position[0] - 1
    elif action == 3:
        new_position[1] = cur_position[1] + 1
        if predictSlip():
            if isXBounded(new_position[0], q_table):
                new_position[0] = cur_position[0] + 1 if cur_position[0] == 0 else cur_position[0] - 1
            else:
                if random.randint(0, 1) == 1:
                    new_position[0] = cur_position[0] + 1
                else:
                    new_position[0] = cur_position[0] - 1
    return new_position


# left right border
def isYBounded(position, q_table):
    if (position == 0) or (position == (len(q_table[0]) - 1)):
        return True
    else:
        return False


# up bottom border
def isXBounded(position, q_table):
    if (position == 0) or (position == (len(q_table) - 1)):
        return True
    else:
        return False


def initTable(q_table):
    for x in range(len(q_table)):
        for y in range(len(q_table[0])):
            if x == 0:  # in the top
                q_table[x][y][0] = None
            if x == (len(q_table) - 1):  # in the bottom
                q_table[x][y][1] = None
            if y == 0:
                q_table[x][y][2] = None
            if y == (len(q_table[0]) - 1):
                q_table[x][y][3] = None
    # for x in range(len(q_table)):
    #     for y in range(len(q_table[0])):
    #         for z in range(len(q_table[x][y][0])):
    #             if q_table[x][y][z] is None:
    #                 continue
    #             action = z
    #             position = [x,y]


def predictAction(cur_position, q_table, weight, maze):
    x = cur_position[0]
    y = cur_position[1]
    actions = q_table[x][y]
    max_qvalue = -sys.maxsize - 1
    max_action = 0
    index = 0
    for a in actions:
        if a is None:
            continue
        feature_vector = getFeatureVector(cur_position, index, maze)
        q_value = feature_vector[0] * weight[0] + feature_vector[1] * weight[1]
        if q_value > max_qvalue:
            max_qvalue = q_value
            max_action = index
        index += 1
    return max_action


def predictSlip():
    slip = random.randint(0, 9)
    if slip < 8:
        return False
    else:
        return True


if __name__ == "__main__":
    getInputs()
