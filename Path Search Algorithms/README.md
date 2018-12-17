# Abstraction

### State space : 

All cities and intersections given in the city-gps.txt and road-segments.txt constitute the state space of the problem.

### Successor function : 

All cities that can be reached from a given city/intersection according to the file road-segments.txt
If algo='astar', we are not putting visited states in the fringe
Please note, we are removing some of the successor states, after considering them as noise in the data.
Below is the definition of noise :

* If Algorithm is Astar and (cost function is distance or time)
   * Remove all road segments where (road distance + 50) is less than crow-fly distance(i.e. a straight line distance between them)
   * Remove all road segments where speed limit is missing or 0(zero).
* Else
   * Remove all road segments where speed limit is missing or 0(zero).

Please note the thought process for choosing such a noise definition: <br/>
We are calculating the straight line distance(crow fly distance) between point A and point B. Let that be X miles. Now
 this distance should always be the least of all the possible road distances between A and B. In other words, all possible
 roads from A to B or B to Ashould always be less than X. But in reality the data is such that there are several such
links/segments which have road distances less than the crow fly distance. But if we remove all of these links as noise,
 for most values of initial and goal cities we would not get any solution. Therefore, we have chosen a buffer of 50 miles,
 so that we get a solution. Due to this, the code might not return optimal solution always, but would at least
 return some solution, which might not be the optimal one due to the data.


### Initial state : 

The start city from the input of the program

### Goal state :  

The end city from the input of the program

### Cost : 

Cost is defined as below:<br/>
* if cost-function(from input)= 'distance'
	* Road Distance from a city/intersection to another city/intersection
* else if cost-function='segment'
	* 1 for each hop from one city/intersection to another city/intersection
* else if cost-function='time'
	* time taken to go from one city/intersection to another city/intersection


### Heuristic Function: 

We have defined our heuristic function as below:<br/>
* If cost-function= 'distance'
  * If Latitude and Longitude is available for the current state(city)
     *  Heuristic of a current state is the crow-fly distance(haversine method) from the current state to the goal state
     *  Haversine method is taken from stack overflow. Link is mentioned below at the function definition
  * Else If Co-ordinates are not available
     *  Get the Co-ordinate of its immediate parent, that would be the crow fly distance from the parent to the goal. Now
        subtract the distance from parent to the current state to get the heuristic of the current state. Here our assumption
        is that the intersection(state without coordinates), parent state and goal state are located on the same straight
        line, and we are going directly toward the goal, when we go from parent to the current state.
* Else If cost-function='time'
     * If Latitude and Longitude is available for the current state(city)
       * Heuristic is defined as the crow-fly distance from state divided by the max of all the speed limits of entire graph +
         road distance of that state from the parent/speed limit of the segment.
     * Else If Co-ordinates are not available
       * Heuristic is defined in the above way for distance,assuming parent, state and goal are at located as a straight line,
       and dividing the whole thing by the speed limit.
* Else if cost-function ='segment'
    * Heuristic is 0 for all states
    
#### Admissibility of the heuristic:
We can say our heuristic is admissible, because we are always underestimating the distance required to reach the goal
 from a current state, since we are taking the straight line distance as a heuristic. When we dont have the coordinates,
we are taking the best possible value that the intersection can have, assuming the parent,goal and the state are at a
 straight line.
 

#Consistency of the heurusitc:
Let's say we move from state A to state B. Cost to move from A to B is c(A,B) Goal state is G. <br/>
Now we can say a heurisitc is consistent if and only if, h(A)<=h(B)+c(A,B), for all values of A and B <br/>
Now let us look at our heuristic. <br/><br/>
For distance: <br/><br/>
h(A) is the straight line distance from A to G<br/>
h(B) is the straight line distance from B to G<br/>
now there are 2 possibilities-<br/>
1.A,B and G are in a straight line, or<br/>
2. A,B and G are points in a triangle<br/>
Possibility 1: A,B and G are in a straight line. Here there can be 3 possibilities:<br/>
   1. A is in between B and G<br/>
   2. B is in between A and G<br/>
   3. G is in between A and B<br/>
        Case 1: since A is in between B and G, h(A) is trivially less than h(B)<br/>
        Case 2: since B is in between A and G, h(B)+c(A,B)>=h(A)(boundary condition, if there is a road in the straight<br/>
        line between A and B, it will be equal, else more)<br/>
        Case 3: since G is in between A and B, then c(A,B) is trivially more than h(A).<br/>
Possibility 2: If A,B and G are in a triangle, then we can say h(A)<=h(B)+c(A,B), if there is a road in the straight<br/>
        line between A and B. Else, c(A,B) will be even more, thus the inequality will always hold.<br/>
For Time:<br/>
 Admissibility of the heuristic:<br/>
 This heuristic is admissible because it always underestimates the time taken from the state to the goal.<br/>
We are taking the least possible distance(crow fly distance) from state to goal and dividing it by the maximum speed limit<br/>
possible for the entire graph, giving us the least distance in which the goal state is reachable(in theory)<br/>
Consistency of the heurisitc:<br/>
To prove consistency we need to show, that for all states h(A)<=h(B)+c(A,B), for all values of A and B<br/>
Let's say we move from state A to state B. Cost to move from A to B is c(A,B) Goal state is G.<br/>
Possibility 1: We know the coordinates of the state<br/>
h(A) is the least time that can be theoritically taken from A to reach the goal.<br/>
h(B) is the least time that can be theoritically taken from B to reach the goal.<br/>
c(A,B) is the time taken to reach from A to B<br/>
Similar case as Possibility 2 of distance, h(A) can never be greater than h(B)+c(A,B), since both h(A) and h(B) are<br/>
 dividing the minimum distance by the maximum speed, it is similar to the triangle inequality of distance, which always<br/>
 holds.<br/>
If we dont know the coordinates, we are taking subtracting the cost from parent to the state from the heuristic of the<br/>
parent, which gives us the theoritical minimum value of the time that we require to go from the state to the goal.<br/>
For Segment:<br/>
 Here heuristic for all states is 1. This is trivially admissible and consistent.

## Algorithm used

Astar, BFS, DFS, IDS, UCS

## Prerequisites

road-segments.txt and city-gps.txt

## How to run

```
./route.py [start-city] [end-city] [routing-algorithm] [cost-function]
```

## Output Format

```
[optimal?] [total-distance-in-miles] [total-time-in-hours] [start-city] [city-1] [city-2] ... [end-city]
```
## Authors

* Ishneet Singh Arora.
* Bivas Maiti.
* Shashank Shekhar.
