# Copyright 2010-2014 Google
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Traveling Salesman Sample.

   This is a sample using the routing library python wrapper to solve a
   Traveling Salesman Problem.
   The description of the problem can be found here:
   http://en.wikipedia.org/wiki/Travelling_salesman_problem.
   The optimization engine uses local search to improve solutions, first
   solutions being generated using a cheapest addition heuristic.
   Optionally one can randomly forbid a set of random connections between nodes
   (forbidden arcs).
"""

import random
import argparse
import math
from ortools.constraint_solver import pywrapcp
# You need to import routing_enums_pb2 after pywrapcp!
from ortools.constraint_solver import routing_enums_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--tsp_size', default=10, type=int,
                    help='Size of Traveling Salesman Problem instance.')
parser.add_argument('--tsp_use_random_matrix', action="store_true",
                    help='Use random cost matrix.')
parser.add_argument('--tsp_random_forbidden_connections', default=0,
                    type=int, help='Number of random forbidden connections.')
parser.add_argument('--tsp_random_seed', default=42, type=int,
                    help='Random seed.')
parser.add_argument('--light_propagation', default=False,
                    type=bool, help='Use light propagation')
parser.add_argument('--num_vehicules', default=0,
                    type=int, help='Number of vehicules on the road.')


# Cost/distance functions and nodes initialization.
class Doors(object):
    """Sets the VRP nodes, which are doors to visit in our case, and calculates cost/distance."""

    def __init__(self):
        """Initial set of doors to visit."""
        #Currently this is a test set
        self.door_list = [
            {"door_id": 55, "latitude": 45.518564, "longitude": -73.583180},  #Tanios Office (Plateau)
            {"door_id": 56, "latitude": 45.517900, "longitude": -73.582380},  #JDA office (Plateau)
            {"door_id": 57, "latitude": 45.478203, "longitude": -73.568707},  #Charlevoix/Centre (Sud-Ouest)
            {"door_id": 64, "latitude": 45.470603, "longitude": -73.568177},  #Caisse/Lasalle (Sud-Ouest)
            {"door_id": 58, "latitude": 45.548690, "longitude": -73.587279},  #1st Avenue/Beaubien (Rosemont)
            {"door_id": 59, "latitude": 45.549423, "longitude": -73.590423},  #Molson/Beaubien (Rosemont)
            {"door_id": 60, "latitude": 45.547474, "longitude": -73.587774},  #Iberville/Bellechasse (Rosemont)
            {"door_id": 61, "latitude": 45.436459, "longitude": -73.633451},  #Danièle/Pauline (Lasalle)
            {"door_id": 62, "latitude": 45.428341, "longitude": -73.629370},  #Dollard/David Boyer (Lasalle)
            {"door_id": 63, "latitude": 45.422917, "longitude": -73.631343},  #Parent/Jeté (Lasalle)

        ]

    def haversine(self, angle):
        """trigonometric formula used in distance"""
        h = math.sin(angle / 2) ** 2
        return h

    def earth_distance(self, lat1, long1, lat2, long2):
        """ Calculates distance between 2 points on Earth"""
        # Note: The formula used in this function is not exact, as it assumes
        # the Earth is a perfect sphere.

        # Mean radius of Earth in meters
        radius_earth = 6371000

        # Convert latitude and longitude to
        # spherical coordinates in radians.
        degrees_to_radians = math.pi / 180.0
        phi1 = lat1 * degrees_to_radians
        phi2 = lat2 * degrees_to_radians
        lambda1 = long1 * degrees_to_radians
        lambda2 = long2 * degrees_to_radians
        dphi = phi2 - phi1
        dlambda = lambda2 - lambda1

        a = self.haversine(dphi) + math.cos(phi1) * math.cos(phi2) * self.haversine(dlambda)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = radius_earth * c
        return int(d)

    def distance(self, node1, node2):
        """Returns the distance between 2 doors (nodes)"""
        #the doors are referenced by their position in the list
        lat1, long1 = self.door_list[node1]["latitude"], self.door_list[node1]["longitude"]
        lat2, long2 = self.door_list[node2]["latitude"], self.door_list[node2]["longitude"]
        return self.earth_distance(lat1, long1, lat2, long2)


class RandomMatrix(object):
    """Random matrix, for when Use Random Matrix = True. This creates random nodes."""

    def __init__(self, size, seed):
        """Initialize random matrix."""

        rand = random.Random()
        rand.seed(seed)
        distance_max = 100
        self.matrix = {}
        for from_node in range(size):
            self.matrix[from_node] = {}
            for to_node in range(size):
                if from_node == to_node:
                    self.matrix[from_node][to_node] = 0
                else:
                    self.matrix[from_node][to_node] = rand.randrange(
                        distance_max)

    def Distance(self, from_node, to_node):
        return self.matrix[from_node][to_node]


def main(args):
    # Create routing model
    if args.tsp_size > 0:
        # TSP of size args.tsp_size
        # Second argument is the number of routes, 1 for each vehicule
        # Nodes are indexed from 0 to parser_tsp_size - 1, by default the start
        # of the route is node 0.
        routing = pywrapcp.RoutingModel(args.tsp_size, args.num_vehicules, 0)

        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        # Setting first solution heuristic (cheapest addition).
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Setting the cost function.
        # Put a callback to the distance accessor here. The callback takes two
        # arguments (the from and to node indices) and returns the distance
        # between these nodes.
        if args.tsp_use_random_matrix:
            matrix = RandomMatrix(args.tsp_size, args.tsp_random_seed)
            matrix_callback = matrix.Distance
            routing.SetArcCostEvaluatorOfAllVehicles(matrix_callback)
        else:
            doors = Doors()
            doors_callback = doors.distance
            routing.SetArcCostEvaluatorOfAllVehicles(doors_callback)

        # Forbid node connections (randomly).
        rand = random.Random()
        rand.seed(args.tsp_random_seed)
        forbidden_connections = 0
        while forbidden_connections < args.tsp_random_forbidden_connections:
            from_node = rand.randrange(args.tsp_size - 1)
            to_node = rand.randrange(args.tsp_size - 1) + 1
            if routing.NextVar(from_node).Contains(to_node):
                print('Forbidding connection ' + str(from_node) + ' -> ' + str(
                    to_node))
                routing.NextVar(from_node).RemoveValue(to_node)
                forbidden_connections += 1

        # Solve, returns a solution if any.
        #    assignment = routing.SolveWithParameters(search_parameters)
        assignment = routing.Solve()
        if assignment:
            # Solution cost.
            print("Total distance: " + str(
                assignment.ObjectiveValue()) + " m\n")
            # Inspect solution.
            # iterate from 0 to routing.vehicles() - 1
            for route_number in range(routing.vehicles()):
                node = routing.Start(route_number)
                route = ''
                route_dist = 0

                while not routing.IsEnd(node):
                    route += str(node) + ' -> '
                    node = assignment.Value(routing.NextVar(node))
                    #route_dist += dist_callback(node_index, node_index_next)

                route += '0'
                #route_dist += dist_callback(node_index, node_index_next)

                print("Route for vehicle " + str(route_number) + ":\n" + route)
                print("Distance of route " + str(route_number) + ": " + str(route_dist) + "\n")

        else:
            print('No solution found.')
    else:
        print('Specify an instance greater than 0.')


if __name__ == '__main__':
    args = parser.parse_args()
    args.tsp_random_seed = 2
    args.tsp_size = 10
    args.tsp_use_random_matrix = False
    args.num_vehicules = 3
    print(args, "\n")
    main(args)