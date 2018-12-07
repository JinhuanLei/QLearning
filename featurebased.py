import math
import os

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
    featurebased_Q_Learning(maze, start_position, 0.9, 0.9, 0.9)


def featurebased_Q_Learning(maze, start_position, learning_rate, policy_randomness, future_discount):
    feature_table = np.zeros([len(maze), len(maze[0]), 4, 2])
    initFeatureTable(feature_table, maze)
    print(feature_table)


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


if __name__ == "__main__":
    getInputs()
