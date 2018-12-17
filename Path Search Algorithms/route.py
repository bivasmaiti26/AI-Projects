#!/bin/python

# put your routing program here!
#!/bin/python
# put your group assignment problem here!

# --------------------------------------abstraction-----------------------------------------------
# state space : All cities and intersections given in the city-gps.txt and road-segments.txt constitute the state space
# of the problem.
##############################################################
# successor function : All cities that can be reached from a given city/intersection according to the file road-segments.txt
#If algo='astar', we are not putting visited states in the fringe
#Please note, we are removing some of the successor states, after considering them as noise in the data.
#Below is the definition of noise.
# If Algorithm is Astar and (cost function is distance or time)
#   Remove all road segments where (road distance + 50) is less than crow-fly distance(i.e. a straight line distance between them)
#   Remove all road segments where speed limit is missing or 0(zero).
#Else
#   Remove all road segments where speed limit is missing or 0(zero).
#
#Please note the thought process for choosing such a noise definition:
#We are calculating the straight line distance(crow fly distance) between point A and point B. Let that be X miles. Now
# this distance should always be the least of all the possible road distances between A and B. In other words, all possible
# roads from A to B or B to Ashould always be less than X. But in reality the data is such that there are several such
# links/segments which have road distances less than the crow fly distance. But if we remove all of these links as noise,
# for most values of initial and goal cities we would not get any solution. Therefore, we have chosen a buffer of 50 miles,
# so that we get a solution. Due to this, the code might not return optimal solution always, but would at least
# return some solution, which might not be the optimal one due to the data.
############################################################################################################################
# initial state : the start city from the input of the program
############################################################################################################################
# goal state :  the end city from the input of the program
############################################################################################################################
# cost : Cost is defined as below:
# if cost-function(from input)= 'distance'
#   Road Distance from a city/intersection to another city/intersection
# else if cost-function='segment'
#   1 for each hop from one city/intersection to another city/intersection
# else if cost-function='time'
#   time taken to go from one city/intersection to another city/intersection
############################################################################################################################
#Heuristic Function: We have defined our heuristic function as below:
#If cost-function= 'distance'
#   If Latitude and Longitude is available for the current state(city)
#       Heuristic of a current state is the crow-fly distance(haversine method) from the current state to the goal state
#       Haversine method is taken from stack overflow. Link is mentioned below at the function definition
#   Else If Co-ordinates are not available
#       Get the Co-ordinate of its immediate parent, that would be the crow fly distance from the parent to the goal. Now
#       subtract the distance from parent to the current state to get the heuristic of the current state. Here our assumption
#       is that the intersection(state without coordinates), parent state and goal state are located on the same straight
#       line, and we are going directly toward the goal, when we go from parent to the current state.
#Else If cost-function='time'
#   If Latitude and Longitude is available for the current state(city)
#       Heuristic is defined as the crow-fly distance from state divided by the max of all the speed limits of entire graph +
#       road distance of that state from the parent/speed limit of the segment.
#   Else If Co-ordinates are not available
#       Heuristic is defined in the above way for distance,assuming parent, state and goal are at located as a straight line,
#       and dividing the whole thing by the speed limit.
#Else if cost-function ='segment'
#   heuristic is 0 for all states
#
#Admissibility of the heuristic:
#We can say our heuristic is admissible, because we are always underestimating the distance required to reach the goal
# from a current state, since we are taking the straight line distance as a heuristic. When we dont have the coordinates,
#we are taking the best possible value that the intersection can have, assuming the parent,goal and the state are at a
# straight line.
#Consistency of the heurusitc:
# Let's say we move from state A to state B. Cost to move from A to B is c(A,B) Goal state is G.
# Now we can say a heurisitc is consistent if and only if, h(A)<=h(B)+c(A,B), for all values of A and B
#Now let us look at our heuristic.
#For distance:
# h(A) is the straight line distance from A to G
# h(B) is the straight line distance from B to G
# now there are 2 possibilities-
# 1.A,B and G are in a straight line, or
# 2. A,B and G are points in a triangle
#Possibility 1: A,B and G are in a straight line. Here there can be 3 possibilities:
#   1. A is in between B and G
#   2. B is in between A and G
#   3. G is in between A and B
#        Case 1: since A is in between B and G, h(A) is trivially less than h(B)
#        Case 2: since B is in between A and G, h(B)+c(A,B)>=h(A)(boundary condition, if there is a road in the straight
#        line between A and B, it will be equal, else more)
#        Case 3: since G is in between A and B, then c(A,B) is trivially more than h(A).
#Possibility 2: If A,B and G are in a triangle, then we can say h(A)<=h(B)+c(A,B), if there is a road in the straight
#        line between A and B. Else, c(A,B) will be even more, thus the inequality will always hold.
#
#For Time:
# Admissibility of the heuristic:
# This heuristic is admissible because it always underestimates the time taken from the state to the goal.
#We are taking the least possible distance(crow fly distance) from state to goal and dividing it by the maximum speed limit
#possible for the entire graph, giving us the least distance in which the goal state is reachable(in theory)
#
#Consistency of the heurisitc:
#To prove consistency we need to show, that for all states h(A)<=h(B)+c(A,B), for all values of A and B
#Let's say we move from state A to state B. Cost to move from A to B is c(A,B) Goal state is G.
#Possibility 1: We know the coordinates of the state
#h(A) is the least time that can be theoritically taken from A to reach the goal.
#h(B) is the least time that can be theoritically taken from B to reach the goal.
#c(A,B) is the time taken to reach from A to B
# Similar case as Possibility 2 of distance, h(A) can never be greater than h(B)+c(A,B), since both h(A) and h(B) are
# dividing the minimum distance by the maximum speed, it is similar to the triangle inequality of distance, which always
# holds.
#If we dont know the coordinates, we are taking subtracting the cost from parent to the state from the heuristic of the
#parent, which gives us the theoritical minimum value of the time that we require to go from the state to the goal.
#
#For Segment:
# Here heuristic for all states is 1. This is trivially admissible and consistent.
############################################################################################################################

import sys
import Queue
from math import radians, cos, sin, asin, sqrt

# returns cost on the basis of choice of user
def cost(state):
    if cost_function == 'distance':
        return state['distance']
    if cost_function == 'segment':
        return 1
    if cost_function == 'time':
        return float('inf') if state['speed_limit'] == 0 else float(state['distance']) / float(state['speed_limit'])
    return 0

# queries the graph against city passed as a state, the graph returns list of
# cities connected to state
# the next city is then passed to cost function to calculate its cost
def successor(city):
    if city in graph:
        return [(next_city["city"], cost(next_city)) for next_city in graph[city]['links']]
    return []

#Check if state is goal city
def is_goal(city):
    return city == end_city


# simple bfs solution
# used python's queue
def solve_bfs(initial_state):
    fringe = Queue.Queue()
    if is_goal(initial_state):
        return initial_state
    fringe.put((initial_state, 0,0,0, initial_state))
    while not (fringe.empty()):
        (state, cost_so_far,time_so_far,distance_so_far, city_so_far) = fringe.get()
        if is_goal(state):
            return isOptimal() + str(distance_so_far) + " " + str(round(time_so_far, 4)) + " " + city_so_far
        for (succ, cost) in successor(state):
            fringe.put((succ, cost_so_far + cost,time_so_far+ gettimedistancebetweensegments(state,succ)[0], distance_so_far+gettimedistancebetweensegments(state,succ)[1], city_so_far + " " + succ))
    return "No Route found!"


# simple dfs solution
# used simple list as a stack
# keeping track of visited nodes using hashmap
def solve_dfs(initial_state):
    visited = {}
    fringe = [(initial_state, 0,0,0, initial_state)]
    while len(fringe) != 0:
        (state, cost_so_far,time_so_far,distance_so_far, city_so_far) = fringe.pop()
        visited[state] = True
        for (succ, cost) in successor(state):
            if is_goal(succ):
                return isOptimal() + str(distance_so_far) + " " + str(round(time_so_far, 4)) + " " + city_so_far
            if not visited.get(succ, False):
                fringe.append((succ, cost_so_far + cost,time_so_far+ gettimedistancebetweensegments(state,succ)[0], distance_so_far+gettimedistancebetweensegments(state,succ)[1], city_so_far + " " + succ))
    return "No Route found!"

#Get the time and distance between 2 states. Time = distance/speed limit
def gettimedistancebetweensegments(state,succ):
    for link in graph[succ]['links']:
        if link['city'] == state:
            return [float(link['distance'])/float(link['speed_limit']),link['distance']]
# each time the fringe pops, the depth traversed counter is incremented, this counter
# is then compared to check if its allowed more depth traversal. If not, it then
# terminates the while loop, else continue exploring the children.
def solve_ids(initial_state):
    depth_threshold = 1
    while True:
        current_depth_explored = 0
        fringe = [(initial_state, 0,0,0, initial_state)]
        while len(fringe) != 0:
            (state, cost_so_far,time_so_far,distance_so_far, city_so_far) = fringe.pop()
            current_depth_explored += 1
            for (succ, cost) in successor(state):
                if is_goal(succ):
                    return isOptimal() + str(distance_so_far) + " " + str(round(time_so_far, 4)) + " " + city_so_far
                if current_depth_explored < depth_threshold:
                    fringe.append((succ, cost_so_far + cost,time_so_far+ gettimedistancebetweensegments(state,succ)[0], distance_so_far+gettimedistancebetweensegments(state,succ)[1], city_so_far + " " + succ))
        depth_threshold += 1
    return "No Route found!"


# uniform cost search
# used python's priorityqueue
# f(s) = c(s), c(s) = cost to reach current state, i.e, each transition costs 1
def solve_ucs(initial_state):
    fringe = Queue.PriorityQueue()
    if is_goal(initial_state):
        return initial_state
    fringe.put((1, (initial_state, 0,0,0, initial_state)))
    while not (fringe.empty()):
        (pri, (state, cost_so_far,time_so_far,distance_so_far, city_so_far)) = fringe.get()
        if is_goal(state):
            return isOptimal() + str(distance_so_far) + " " + str(round(time_so_far, 4)) + " " + city_so_far
        for (succ, cost) in successor(state):
            fringe.put((cost_so_far + 1,
                        (succ, cost_so_far + 1,time_so_far+ gettimedistancebetweensegments(state,succ)[0], distance_so_far+gettimedistancebetweensegments(state,succ)[1], city_so_far + " " + succ)))
    return "No Route found!"


#  A*
# used python's priorityqueue
# f(s) = c(s), c(s) = cost to reach current state, i.e, distance
def solve_Astar(initial_state):
    visited = {}
    fringe = Queue.PriorityQueue()
    if is_goal(initial_state):
        return initial_state
    #heuristic = calc_heuristic(initial_state)
    fringe.put((0, (initial_state, 0,0,0, initial_state, 0)))
    while not (fringe.empty()):
        (pri, (state, cost_so_far,time_so_far,distance_so_far, city_so_far, hu)) = fringe.get()
        visited[state] = True
        if is_goal(state):
            return isOptimal() + str(distance_so_far) + " " + str(round(time_so_far, 4)) + " " + city_so_far
        for (succ, cost) in successor(state):
            heuristic = calc_heuristic(succ,cost,hu,state)
            if (not(visited.get(succ,False))) or cost_function=='segment':
                fringe.put((cost_so_far + cost + heuristic, (succ, cost_so_far + cost, time_so_far+ gettimedistancebetweensegments(state,succ)[0], distance_so_far+gettimedistancebetweensegments(state,succ)[1],city_so_far + " " + succ, heuristic)))
                visited[succ]=True
    return "No Route found!"


# euclidean distance as a heuristic
# if a node doesn't have a coordinate then take h(s) = 0, that means
# consider only cost of reaching that node as f(s), always underestimate the h(s)
# if a state is visited then h(s) is infinity
# need to check for inconsistency
def calc_heuristic(state,cost,parenthu,parent):
    if cost_function == 'distance' :
        goal_coordinates = graph[end_city]['coordinates']
        coordinates = graph[state]['coordinates']
        if len(coordinates) == 0:
            return parenthu-cost
        else:
            #print(state+":"+str(haversine(coordinates, goal_coordinates)))
            return haversine(coordinates, goal_coordinates)
    if cost_function == 'time' :
        for link in  graph[state]['links']:
            if link['city']==parent:
                speed_limit=link['speed_limit']
        goal_coordinates = graph[end_city]['coordinates']
        coordinates = graph[state]['coordinates']
        if len(coordinates) == 0:
            return parenthu-cost
        else:
            return float(cost)+ float(haversine(coordinates, goal_coordinates))/float(MAX_SPEED_LIMIT)
    if cost_function == 'segment' :
        return 0

#https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(coord1, coord2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1=coord1[1]
    lon2=coord2[1]
    lat1=coord1[0]
    lat2=coord2[0]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km/1.6

def calulate_eucledian(coord1, coord2):
    return sqrt(pow((coord2[0] - coord1[0])*68, 2) + pow((coord2[1] - coord1[1])*69, 2))


graph = {}

############################ Building Data Structure for graph#########################
# note : use print_graph() to see how graph for this problem works
# this function will create a file called graph.txt in the current directory.
def readRoadSegments():
    global MAX_SPEED_LIMIT
    with open("road-segments.txt", "r") as f:
        for line in f:
            strings_in_line = line.split()
            from_city = strings_in_line[0]
            to_city = strings_in_line[1]
            dist = int(strings_in_line[2])

            # if there is no speed limit
            if len(strings_in_line) < 5:
                #continue
                #bivas change
                #speed_limit = float('inf')
                speed_limit = 0
                hname = strings_in_line[3]
            else :
                speed_limit = int(strings_in_line[3])
                if (speed_limit > MAX_SPEED_LIMIT):
                    MAX_SPEED_LIMIT = speed_limit
                hname = strings_in_line[4]
            link = {"city": to_city, "distance": dist, "speed_limit": speed_limit, "highway_name": hname}
            if (from_city in graph):
                graph[from_city]['links'].append(link)
            else:
                graph[from_city] = {'links': [link], 'coordinates': []}

            # now handle bidirection

            # swap to_city and from_city
            temp = to_city
            to_city = from_city
            from_city = temp

            link = {"city": to_city, "distance": dist, "speed_limit": speed_limit, "highway_name": hname}
            if (from_city in graph):
                graph[from_city]['links'].append(link)
            else:
                graph[from_city] = {'links': [link], 'coordinates': []}

#If the algorithm is returning an optimal solution, return yes, else no
def isOptimal():
    if algo=="astar"  or algo=="uniform":
        return "yes "
    elif algo=="ids" or algo=="bfs":
        if cost_function=="segment" :
            return "yes "
        else:
            return "no "
    elif algo=="dfs":
        return "no "

#Read the city-gps.txt file
def readCityGps():
    with open("city-gps.txt", "r") as f:
        for line in f:
            strings_in_line = line.split()
            # if city is missing from GPS list, its coordinate is taken as empty
            for city in graph:
                if city == strings_in_line[0]:
                    graph[city]['coordinates'].append(float(strings_in_line[1]))
                    graph[city]['coordinates'].append(float(strings_in_line[2]))

#Remove noise from the data gathered
def removeNoise():
    badlinks = []
    for city in graph:
        coordinates = graph[city]['coordinates']
        if len(coordinates) > 0:
            for link in graph[city]['links']:
                coordinateslink=graph[link['city']]['coordinates']
                if len(coordinateslink) > 0:
                    disteuc=calulate_eucledian(coordinates,coordinateslink)
                    if(link['distance']+50<disteuc):
                        badlinks.append( city + " | "+link['city']+"|D")
                        if (algo=='astar' and (cost_function != 'segment')):
                            graph[city]['links'].remove(link)
                        continue
    for city in graph:
        for link in graph[city]['links']:
            if link['speed_limit']==0:
                badlinks.append(city + " | "+link['city']+"|S")
                graph[city]['links'].remove(link)
    file=open("badlinks.txt","w")
    for line in badlinks:
        file.write(line+"\n")
    file.close()

#Print graph in a file (for debugging purposes)
def print_graph():
    msg = "{\n"
    file = open("graph.txt", "w")
    for city in graph:
        msg = msg + "\n\t\t" + city + " : { links : [ \n"
        for link in graph[city]['links']:
            msg = msg + "\n\t\t{\n"
            msg = msg + "\t\t\tcity : " + link['city'] + ",\n\t\t\tdistance : " + str(
                link['distance']) + ",\n\t\t\tspeed_limit : " + str(link['speed_limit']) + ",\n\t\t\thighway_name : " + \
                  link['highway_name'] + "\n\t\t},"
        msg = msg[0:len(msg) - 1] + "],\n"
        if len(graph[city]['coordinates']) != 0:
            msg = msg + "\t\tcoordinates : (" + str(graph[city]['coordinates'][0]) + ", " + str(
                graph[city]['coordinates'][1]) + ")"
        else:
            msg = msg + "\t\tcoordinates : ()"
        msg = msg + "\n\t\t},"
    msg = msg[0:len(msg) - 1]
    msg = msg + "\n}"
    file.write(msg)
    file.close()

MAX_SPEED_LIMIT=0
readRoadSegments()
readCityGps()

#print_graph()
start_city = sys.argv[1]
end_city = sys.argv[2]
algo = sys.argv[3]
cost_function = 'distance' if sys.argv[4] == 'distance' else 'segment' if sys.argv[4] == 'segment' else 'time' if \
sys.argv[4] == 'time' else 'unknown'
removeNoise()
#print(graph)
# print(graph['Hot_Springs,_Arkansas'][2].getCity())
# print("uniform cost - actual cost in distance - start city city1 city2 ... city n goal city")
if algo=='astar':
    print(solve_Astar(start_city))
elif algo=='bfs':
    print(solve_bfs(start_city))
elif algo=='uniform':
    print(solve_ucs(start_city))
elif algo=='ids':
    print(solve_ids(start_city))
elif algo=='dfs':
    print(solve_dfs(start_city))
else:
    print("Wrong Arguements")

