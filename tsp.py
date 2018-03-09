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
import numpy as np
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
    """Manhattan distance"""
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
                # Define the distance from the depot to any node to be 0.
                if from_node == depot or to_node == depot:
                    self.matrix[from_node][to_node] = 0
                else:
                    x1 = locations[from_node][0]
                    y1 = locations[from_node][1]
                    x2 = locations[to_node][0]
                    y2 = locations[to_node][1]
                    self.matrix[from_node][to_node] = earth_distance(x1, y1, x2, y2)

    def distance(self, from_node, to_node):
        return int(self.matrix[from_node][to_node])


# Time callback
class CreateTimeCallback(object):
    """Create callback to get total times between locations."""

    ##the init function is probably useless and I think we just need the time#
    #we define that all locations (meetings) last 60 minutes
    def __init__(self):
        self.matrix = 60

    def time(self, from_node, to_node):
        return self.matrix


def main():
    # import data
    data = import_data()
    locations = data[0]
    schedule = data[1]
    skill = data[2]
    interest = data[3]
    node_to_door = data[4]
    route_to_salesman = data[5]

    num_locations = len(locations)
    num_vehicles = len(route_to_salesman)

    # depot is Home, the start point of each route for first meeting and end
    # point after last meeting, it is defined as distance 0 to any other node
    # In other words, Home is the salesman Home, as in: he is off the clock
    depot = 0

    # Create routing model.
    if num_locations > 0:
        routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

        # Callback to the distance function.
        dist_between_locations = CreateDistanceCallback(locations)
        dist_callback = dist_between_locations.distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

        # Put a callback to the time. No argument required since we define
        #that all meetings last for 60 minutes
        times_at_locations = CreateTimeCallback()
        times_callback = times_at_locations.time

        # Add a dimension for time.
        time_slack = 15
        time_d_name = "Time"
        vehicle_capacity = int(max(schedule) + 60)  # the 60 is to return Home
        routing.AddDimension(times_callback, time_slack, vehicle_capacity, True, time_d_name)

        #Add the time windows constraint, which is the meeting schedule
        time_dimension = routing.GetDimensionOrDie(time_d_name)
        for location in range(1, num_locations):
            start = int(schedule[location] - time_slack)
            end = int(schedule[location] + time_slack)
            time_dimension.CumulVar(location).SetRange(start, end)

        # Solve, displays a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            # Display solution.
            # Solution cost.
            print("Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\n")

            for vehicle_nbr in range(num_vehicles):
                index = routing.Start(vehicle_nbr)
                index_next = assignment.Value(routing.NextVar(index))
                route = ''
                route_dist = 0
                route_times = ''

                while not routing.IsEnd(index_next):
                    node_index = routing.IndexToNode(index)
                    node_index_next = routing.IndexToNode(index_next)
                    route += str(node_index) + " -> "
                    # Add the distance to the next node.
                    route_dist += dist_callback(node_index, node_index_next)
                    # Add time-delayed based schedule
                    route_times += str(schedule[node_index]) + " -> "
                    index = index_next
                    index_next = assignment.Value(routing.NextVar(index))

                node_index = routing.IndexToNode(index)
                node_index_next = routing.IndexToNode(index_next)
                route += str(node_index) + " -> " + str(node_index_next)
                route_times += str(schedule[node_index]) + " -> " + str(vehicle_capacity)
                route_dist += dist_callback(node_index, node_index_next)
                print("Route for Salesman " + str(vehicle_nbr) + ":\n" + route)
                print(route_times)
                print("Distance of Route " + str(vehicle_nbr) + ": " + str(route_dist))

        else:
            print('No solution found.')
    else:
        print('Specify an instance greater than 0.')


def import_data():
    """imports data from specified file"""
    df_salesmen = pd.read_csv("tsp-python-import - salesmen.csv")
    df_locations = pd.read_csv("tsp-python-import - locations.csv")
    df_meetings = pd.read_csv("tsp-python-import - meetings.csv",
                              parse_dates=[1], infer_datetime_format=True)

    # add the coordinate info to meetings
    df_meetings = pd.merge(df_meetings, df_locations, how="left",
                           left_on="meet_location", right_on="door_id",
                           sort=False, validate="1:1")

    # convert the schedule from datetime value to a chronometered value in
    # minutes starting from the depot at 0, thus first meeting starts at 60
    # ASSUMPTION: ALL MEETINGS LAST 1 HOUR AND ARE SCHEDULED ACCORDINGLY
    schedule = df_meetings["meet_time"].values
    schedule = (schedule - np.min(schedule)).astype("timedelta64[m]")
    schedule = schedule.astype("int64") + 60
    schedule = np.concatenate(([0], schedule))

    # matrix, each row is a pair of coordinates
    location = df_meetings[["door_latitude", "door_longitude"]].values
    # add Home coordinates
    location = np.concatenate(([[0, 0]], location), axis=0)

    # maps a node to a door_id
    node_to_door = df_meetings["meet_location"].values
    # add the Home node
    node_to_door = np.concatenate(([0], node_to_door))

    # map routes to salesman
    route_to_salesman = df_salesmen["salesman_name"].values

    # skill is the quality of the salesman, interest is the quality of the meeting
    # better salesman gets better meeting
    skill = df_salesmen["salesman_skill"].values
    interest = df_meetings["meet_interest"].values

    return [location, schedule, skill, interest, node_to_door,
            route_to_salesman]


if __name__ == '__main__':
    main()
