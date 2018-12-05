import os
import random
import sys

import numpy

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
    Q_Learning(maze, start_position, 0.9, 0.9)


def Q_Learning(maze, start_position, learning_rate, policy_randomness):
    # 0 up 1 down 2 left 3 right
    q_table = numpy.zeros([len(maze), len(maze[0]), 4])
    initTable(q_table)
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
            action = predictAction(cur_position, q_table)
            # Observe next state
            new_position = getNewPosition(cur_position, action, q_table)
            reward = getReward(new_position, maze)
            updateQValue(cur_position, new_position, action, q_table, learning_rate, policy_randomness, reward)
            cur_position = new_position
            steps += 1
            life = isContinue(new_position, maze, steps)


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



def drawResult():
    pass


def updateQValue(cur_position, new_position, action, q_table, learning_rate, policy_randomness, reward):
    q_value = q_table[cur_position[0]][cur_position[1]][action]
    actions = q_table[new_position[0]][new_position[1]]
    max_qvalue = findMax(actions)
    q_value = q_value + learning_rate * (reward + policy_randomness * max_qvalue - q_value)
    q_table[cur_position[0]][cur_position[1]][action] = q_value


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
    elif steps == (len(maze) * len(maze[0])):
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
            if isYBounded(cur_position[1], q_table):
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
            if isYBounded(cur_position[1], q_table):
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
            if isXBounded(cur_position[0], q_table):
                new_position[0] = cur_position[0] + 1 if cur_position[0] == 0 else cur_position[0] - 1
            else:
                if random.randint(0, 1) == 1:
                    new_position[0] = cur_position[0] + 1
                else:
                    new_position[0] = cur_position[0] - 1
    elif action == 3:
        new_position[1] = cur_position[1] + 1
        if predictSlip():
            if isXBounded(cur_position[0], q_table):
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


def predictAction(cur_position, q_table):
    zero_actions = []
    x = cur_position[0]
    y = cur_position[1]
    # if(x)
    actions = q_table[x][y]
    maxVal = -sys.maxsize - 1
    maxAction = 0
    index = 0
    for a in actions:
        if a is None:
            continue
        if a > maxVal:
            maxVal = a
            maxAction = index
        if a == 0:
            zero_actions.append(index)
        index += 1
    if len(zero_actions) == 0:
        return maxAction
    else:
        # random choose an action
        action = random.randint(0, len(zero_actions) - 1)
        return zero_actions[action]


def predictSlip():
    slip = random.randint(0, 9)
    if slip < 8:
        return False
    else:
        return True


if __name__ == "__main__":
    getInputs()
