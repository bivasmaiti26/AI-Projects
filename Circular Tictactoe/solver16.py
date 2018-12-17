#!/bin/python
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018
## Cost : cost is cost of moves so far that is total number of moves performed to reach your goal state.

# We have used multiple hueristics which is givig optimal solutions upto board 9 and we are getting total moves as 16 for board 12
# List of Hueristic we have tried are-

# We have implemented Circular Hueristic and we are getting the output for board 12 in 16 moves.
# We have also tried to implement linear conflicts but the output remains the same.
# In Linear Conflict if the 2 values are in its right row or column but they are switched we add a a #constant value for
# this cause actauuly they wix`ll take the count more than the actual manhattan distance.
# We have also tried to implement min of the right and left moves for each row and min of up and down for each columns but
#  still there was no change in  solutions
# We have also tried to implement max of the right and left moves for each row and max of up and down for each columns
# and then min of the 2 max but in this case we arent getting the #output of board 12 while all other board outcomes are optimum
# We have also tried using Permutation inversion as a hueristic where the outout is coming optimum till board 9 but not for board 12
# We have also tried to different combinations of manhattan distance , linear conflicts, misplaced tiles but we were not
# getting an optimum solution.
# Thus we have selected a heurisitc which calculates the difference of moves of the pieces of a board.
# This calculates the moves required for each piece of the board, and which move is required to go to the correct position.
# Moves can consist of 4 values for each direction :+1,+2,-1 and -2
# Directions are be difined as Horiziontal or Vertical
# For Horiziontal, a cost of +i means, for that peice in the board, we can say it will go to its correct position in the
# column if we move the row 1 step right.
# Similarly for vertical, +i means we need to make downward move i times to get the piece into its correct row.
# So each of the pieces will have a cost attached to it, i.e. an array of length 2, for each piece in the board.
# Now we are checking for each row and each column the number or differences of moves for each of the pieces. This will
# give us 2 arrays, diffcol and diffrow. The heuristic is defined as:
# return (max(diffcol)+sum(diffrow)-4)/2
import Queue
import sys
import string


# shift a specified row left (1) or right (-1)
def shift_row(state, row, dir):
    change_row = state[(row * 4):(row * 4 + 4)]
    return (state[:(row * 4)] + change_row[-dir:] + change_row[:-dir] + state[(row * 4 + 4):],
            ("L" if dir == -1 else "R") + str(row + 1), 1)


# shift a specified col up (1) or down (-1)
def shift_col(state, col, dir):
    change_col = state[col::4]
    s = list(state)
    s[col::4] = change_col[-dir:] + change_col[:-dir]
    return (tuple(s), ("U" if dir == -1 else "D") + str(col + 1), 1)


# pretty-print board state
def print_board(row):
    for j in range(0, 16, 4):
        print '%3d %3d %3d %3d' % (row[j:(j + 4)])


# return a list of possible successor states
def successors(state):
    return [shift_row(state, i, d) for i in range(0, 4) for d in (1, -1)] + [shift_col(state, i, d) for i in range(0, 4)
                                                                             for d in (1, -1)]


# just reverse the direction of a move name, i.e. U3 -> D3
def reverse_move(state):
    return state.translate(string.maketrans("UDLR", "DURL"))


# check if we've reached the goal
def is_goal(state):
    return sorted(state) == list(state)


# The solver! - using BFS right now
def solve(initial_board):
    fringe = [(initial_board, "")]
    while len(fringe) > 0:
        (state, route_so_far) = fringe.pop()
        for (succ, move) in successors(state):
            if is_goal(succ):
                return (route_so_far + " " + move)
            fringe.insert(0, (succ, route_so_far + " " + move))
    return False


# The solver! - using Priority Queue and heuristic.
# using calculateHeuristic(state) function which calculates heuristics at each state.
# f(s) = g(s) + h(s)
# g(s) = cost of next move, h(s) = number of misplaced tiles.
# cost of next move is taken as length of routes_so_far
counter = 0


def solve(initial_board):
    if is_goal(initial_board):
        return ""
    fringe = Queue.PriorityQueue()
    fringe.put((0, (initial_board, "", 0)))
    visited = []
    while not (fringe.empty()):
        (pri, (state, route_so_far, cost_so_far)) = fringe.get()
        if state not in visited:
            visited.append(state)
        if is_goal(state):
            return (route_so_far)
        for (succ, move, cost) in successors(state):
            route_so_far_curr_state = len((route_so_far + " " + move).split())
            clean_fringe_flag = True
            if succ not in visited:
                fringe.put((cost_so_far + 1 + calchuristic_diffmoves(succ),
                            (succ, route_so_far + " " + move, cost_so_far + 1)))
    return False


# h(s) = number of cells which are misplaced
def calculateHeuristic_misplacedtiles(state):
    counter = 0
    for i in range(0, 16):
        if state[i] != i + 1:
            counter += 1
    return counter


def calchuristic_diffmoves(state):
    minmoves = []
    for i in range(0, 16):
        pos = i
        goalpos = state[i] - 1
        rowmoves = goalpos / 4 - pos / 4
        if rowmoves == 3:
            rowmoves = -1
        if rowmoves == - 3:
            rowmoves = 1
        colmoves = goalpos % 4 - pos % 4
        if colmoves == 3:
            colmoves = -1
        if colmoves == -3:
            colmoves = 1
        minmoves.append([rowmoves, colmoves])
    diffrow = []
    diffcol = []
    for j in range(0, 16, 4):
        rowminmoves = []
        for i in range(0, 4):
            if (minmoves[i + j][1] not in rowminmoves):
                rowminmoves.append(minmoves[i + j][1])
        diffrow.append(len(rowminmoves))
    for i in range(0, 4):
        colminmoves = []
        for j in range(0, 4):
            if (minmoves[i + 4 * j][0] not in colminmoves):
                colminmoves.append(minmoves[i + 4 * j][0])
        diffcol.append(len(colminmoves))
    return (max(diffcol) + sum(diffrow) - 4) / 2


# circular manhattan distance heuristic


def calculate_heuristic_circular_manhattan(state):
    goal_position_indices = {16: (3, 3),
                             15: (3, 2),
                             14: (3, 1),
                             13: (3, 0),
                             12: (2, 3),
                             11: (2, 2),
                             10: (2, 1),
                             9: (2, 0),
                             8: (1, 3),
                             7: (1, 2),
                             6: (1, 1),
                             5: (1, 0),
                             4: (0, 3),
                             3: (0, 2),
                             2: (0, 1),
                             1: (0, 0)}
    lienar_to_2d_mapper = {15: (3, 3),
                           14: (3, 2),
                           13: (3, 1),
                           12: (3, 0),
                           11: (2, 3),
                           10: (2, 2),
                           9: (2, 1),
                           8: (2, 0),
                           7: (1, 3),
                           6: (1, 2),
                           5: (1, 1),
                           4: (1, 0),
                           3: (0, 3),
                           2: (0, 2),
                           1: (0, 1),
                           0: (0, 0)}
    circular_manhattan_mapper = {3: 1, 1: 1, 2: 2, 0: 0}
    board_state = []

    sum_of_manhattan_dist = 0
    for i in range(0, 16):
        dx = abs(goal_position_indices[state[i]][0] - lienar_to_2d_mapper[i][0])
        dy = abs(goal_position_indices[state[i]][1] - lienar_to_2d_mapper[i][1])
        dx = circular_manhattan_mapper[dx]
        dy = circular_manhattan_mapper[dy]
        sum_of_manhattan_dist += dx + dy
    rem = sum_of_manhattan_dist % 4
    sum_of_manhattan_dist = sum_of_manhattan_dist / 4 + rem
    return sum_of_manhattan_dist


# permutation inversion
def inversion(state):
    sum = 0
    for i in range(0, 16):
        j = i + 1
        while j < 16:

            if state[i] > state[j]:
                sum += 1
            j += 1
    return sum


# Heuristic-Linear Conflict for Rows
def linearr(state):
    count = 0
    for i in range(0, 16):
        j = i
        if i < 3:
            if (i + 1) in goal_state[0:4]:
                if state[i] in goal_state[0:4]:
                    if (state[i] - goal_state[i]) > 0:
                        count += 1
        if i > 3 and i < 7:
            if (i + 1) in goal_state[4:8]:
                if state[i] in goal_state[4:8]:
                    if (state[i] - goal_state[i]) > 0:
                        count += 1
        if i > 7 and i < 11:
            if (i + 1) in goal_state[8:12]:
                if state[i] in goal_state[8:12]:
                    if (state[i] - goal_state[i]) > 0:
                        count += 1
        if i > 11 and i < 15:
            if (i + 1) in goal_state[12:16]:
                if state[i] in goal_state[12:16]:
                    if (state[i] - goal_state[i]) > 0:
                        count += 1
        if (i == 3 or i == 7 or i == 11 or i == 15):
            if ((state[i] - goal_state[0]) > 0):
                count += 1

    return (count)


# linear conflict of cols
def linearc(state):
    board_state = []
    for i in range(0, 4):
        board_state.append(list(state[(i * 4):(i * 4 + 4)]))
    board_goal = []
    for i in range(0, 4):
        board_goal.append(list(goal_state[(i * 4):(i * 4 + 4)]))

    temps = []
    tempg = []
    j = 0
    count = 0
    while j < 16:
        k = int(j / 4)
        for col in board_state:
            temps.append(col[k])
        for col in board_goal:
            tempg.append(col[k])
        for i in range(0, 4):
            if temps[i] in tempg[0:4]:
                if (temps[i] - tempg[i]) > 0:
                    count += 1
        j += 4
        temps.clear()
        tempg.clear()
    return (count)


# test cases
start_state = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        start_state += [int(i) for i in line.split()]

if len(start_state) != 16:
    print "Error: couldn't parse start state file"

print "Start state: "
print_board(tuple(start_state))
goal_state = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
print "Solving..."
route = solve(tuple(start_state))

print "Solution found in " + str(len(route) / 3) + " moves:" + "\n" + route
