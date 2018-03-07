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

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import pandas as pd
import math


def haversine(angle):
    """trigonometric formula used in distance"""
    h = math.sin(angle / 2) ** 2
    return h


def earth_distance(lat1, long1, lat2, long2):
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

    a = haversine(dphi) + math.cos(phi1) * math.cos(phi2) * haversine(dlambda)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius_earth * c
    return int(d)


def manhattan_distance(x1, y1, x2, y2):
    # Manhattan distance
    dist = abs(x1 - x2) + abs(y1 - y2)
    return int(dist)


class CreateDistanceCallback(object):
    """Create callback to calculate distances between points."""

    def __init__(self, locations):
        """Initialize distance array."""
        size = len(locations)
        depot = 0
        self.matrix = {}

        for from_node in range(size):
            self.matrix[from_node] = {}
            for to_node in range(size):
                if from_node == depot or to_node == depot:
                    # Define the distance from the depot to any node to be 0.
                    self.matrix[from_node][to_node] = 0
                else:
                    x1 = locations[from_node][0]
                    y1 = locations[from_node][1]
                    x2 = locations[to_node][0]
                    y2 = locations[to_node][1]
                    self.matrix[from_node][to_node] = earth_distance(x1, y1, x2, y2)

    def Distance(self, from_node, to_node):
        return int(self.matrix[from_node][to_node])


# Demand callback
class CreateDemandCallback(object):
    """Create callback to get demands at each location."""

    def __init__(self, demands):
        self.matrix = demands

    def Demand(self, from_node, to_node):
        return self.matrix[from_node]


def main():
    # Create the data.
    data = create_data_array()
    locations = data[0]
    demands = data[1]
    area = data[2]
    num_locations = len(locations)
    depot = 0  # The depot is the start and end point of each route.
    num_vehicles = 3

    # Create routing model.
    if num_locations > 0:
        routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

        # Callback to the distance function.
        dist_between_locations = CreateDistanceCallback(locations)
        dist_callback = dist_between_locations.Distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

        # Put a callback to the demands.
        demands_at_locations = CreateDemandCallback(demands)
        demands_callback = demands_at_locations.Demand

        # Add a dimension for demand.
        slack_max = 0
        vehicle_capacity = 100
        fix_start_cumul_to_zero = True
        demand = "Interest"
        routing.AddDimension(demands_callback, slack_max, vehicle_capacity,
                             fix_start_cumul_to_zero, demand)

        # Solve, displays a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            # Display solution.
            # Solution cost.
            print("Total distance of all routes: " + str(
                assignment.ObjectiveValue()) + "\n")

            for vehicle_nbr in range(num_vehicles):
                index = routing.Start(vehicle_nbr)
                index_next = assignment.Value(routing.NextVar(index))
                route = ''
                route_dist = 0
                route_demand = 0
                route_area = ''

                while not routing.IsEnd(index_next):
                    node_index = routing.IndexToNode(index)
                    node_index_next = routing.IndexToNode(index_next)
                    route += str(node_index) + " -> "
                    route_area += str(area[node_index]) + " -> "
                    # Add the distance to the next node.
                    route_dist += dist_callback(node_index, node_index_next)
                    # Add demand.
                    route_demand += demands[node_index_next]
                    index = index_next
                    index_next = assignment.Value(routing.NextVar(index))

                node_index = routing.IndexToNode(index)
                node_index_next = routing.IndexToNode(index_next)
                route += str(node_index) + " -> " + str(node_index_next)
                route_area += str(area[node_index]) + " -> " + str(area[node_index_next])
                route_dist += dist_callback(node_index, node_index_next)
                print("Route for Salesman " + str(vehicle_nbr) + ":\n" + route)
                print(route_area)
                print("Distance of Route " + str(vehicle_nbr) + ": " + str(
                    route_dist))
                print(demand +" met by Salesman " + str(vehicle_nbr) + ": " + str(
                    route_demand) + "\n")
        else:
            print('No solution found.')
    else:
        print('Specify an instance greater than 0.')


def create_data_array():
    # this is testing data
    raw_data = [
        [55, 45.518564, -73.583180, 7, "Saint-Laurent/Marianne", "Plateau"],
        [56, 45.517900, -73.582380, 8, "Saint-Laurent/Rachel", "Plateau"],
        [57, 45.478203, -73.568707, 1, "Charlevoix/Centre", "Sud-Ouest"],
        [64, 45.470603, -73.568177, 3, "Caisse/Lasalle", "Sud-Ouest"],
        [58, 45.548690, -73.587279, 2, "1st Avenue/Beaubien", "Rosemont"],
        [59, 45.549423, -73.590423, 5, "Molson/Beaubien", "Rosemont"],
        [60, 45.547474, -73.587774, 6, "Iberville/Bellechasse", "Rosemont"],
        [61, 45.436459, -73.633451, 5, "Danièle/Pauline", "Lasalle"],
        [62, 45.428341, -73.629370, 9, "Dollard/David Boyer", "Lasalle"],
        [63, 45.422917, -73.631343, 4, "Parent/Jeté", "Lasalle"],
    ]

    #transform the test data into Pandas DataFrame because eventually, we will  import SQL from Pandas
    data = pd.DataFrame(raw_data, columns=["door_id", "latitude", "longitude", "interest", "streets", "area"])

    #adding Home: a dummy depot with 0 distance to any node, since we don't
    # require the salesman to return to a depot
    # consider Home as the salesman clocking in and out of his route
    home = pd.DataFrame([[0, 0, 0, 0, "none/none", "Home"]], columns=["door_id", "latitude", "longitude", "interest", "streets", "area"])
    data = pd.concat([home, data], axis=0, ignore_index=True)

    #formatted data we will be using
    location = pd.concat([data["latitude"], data["longitude"]], axis=1)
    location = location.values
    interest = data["interest"].values
    area = data["streets"].values
    return [location, interest, area]


if __name__ == '__main__':
    main()