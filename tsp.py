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


class CreateDistanceCallback(object):
    """Create callback to calculate distances between points, weighted or not."""

    def __init__(self, locations, interests, skills, weight_strength=0):
        """Initialize distance array.
        If you want the weight, you need to supply all 3 parameters
        interest, skill, and weight_strength. For quickly switching between
        raw (unweighted) and weighted distance, just set weight_strength to 0."""
        size = len(locations)
        num_vehicle = len(skills)
        depot = 0
        self.matrix = {}
        for vehicle in range(num_vehicle):
            self.matrix[vehicle] = {}
            for from_node in range(size):
                self.matrix[vehicle][from_node] = {}
                for to_node in range(size):
                    self.matrix[vehicle][from_node][to_node] = {}
                    weight = 1 + weight_strength * abs(interests[to_node] - skills[vehicle])
                    # Define the distance from the depot to any node to be 0.
                    if from_node == depot or to_node == depot:
                        self.matrix[vehicle][from_node][to_node] = 0
                    else:
                        x1 = locations[from_node][0]
                        y1 = locations[from_node][1]
                        x2 = locations[to_node][0]
                        y2 = locations[to_node][1]
                        self.matrix[vehicle][from_node][to_node] = int(self.earth_distance(x1, y1, x2, y2) * weight)


    def distance(self, from_node, to_node):
        return self.matrix[0][from_node][to_node]

    def distance_v(self, vehicle):
        return lambda from_node, to_node: self.matrix[vehicle][from_node][to_node]

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
        return d

    def manhattan_distance(self, x1, y1, x2, y2):
        """Manhattan distance"""
        dist = abs(x1 - x2) + abs(y1 - y2)
        return dist


class CreateTimeCallback(object):
    """Create callback to get total times between locations."""

    #ASSUMPTION: all meetings last the same (in minutes)
    def __init__(self, meet_length):
        self.time_matrix = int(meet_length)

    def time(self, from_node, to_node):
        return self.time_matrix


def main():
    # import data
    data = import_data(meet_length=60)
    locations = data[0]
    chronometer = data[1]
    skills = data[2]
    interests = data[3]
    node_to_door = data[4]
    route_to_salesman = data[5]
    node_to_time = data[6]

    meet_length = 60 #minutes
    num_locations = len(locations)
    num_vehicles = len(skills)

    # depot is Home, the start point of each route for first meeting and end
    # point after last meeting, it is defined as distance 0 to any other node
    # In other words, Home is the salesman Home, as in: he is off the clock
    depot = 0

    # Create routing model.
    if num_locations > 0:
        routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

        # Callback to the distance function.
        dist_between_locations = CreateDistanceCallback(locations, interests, skills, weight_strength=2)

        #adding the cost function for each vehicle seperately since each
        #salesman has a different skill that we will optimize for
        dist_callbacks = []
        for vehicle in range(num_vehicles):
            dist_callbacks.append(dist_between_locations.distance_v(vehicle))

        for vehicle in range(num_vehicles):
            routing.SetArcCostEvaluatorOfVehicle(dist_callbacks[vehicle], vehicle)

        # Put a callback to the time.
        times_at_locations = CreateTimeCallback(meet_length)
        times_callback = times_at_locations.time

        # Add a dimension for time.
        time_slack = int(meet_length / 4)
        time_d_name = "Time"
        total_work_time = int(max(chronometer) + meet_length)  #enough to return Home
        routing.AddDimension(times_callback, time_slack, total_work_time, True, time_d_name)

        #Add the time windows constraint, which is the meeting schedule
        time_dimension = routing.GetDimensionOrDie(time_d_name)
        for location in range(1, num_locations):
            start = int(chronometer[location] - time_slack)
            end = int(chronometer[location] + time_slack)
            time_dimension.CumulVar(location).SetRange(start, end)

        # Solve, displays a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            # Display solution.
            # Solution cost.
            print("Total (weighted) distance of all routes: " + str(assignment.ObjectiveValue()))

            for vehicle_nbr in range(num_vehicles):
                index = routing.Start(vehicle_nbr)
                index_next = assignment.Value(routing.NextVar(index))
                route = 'Nodes: '
                route_dist = 0
                route_schedule = 'Schedule: '
                route_door = "Door ID: "
                route_interest = "Meet Quality:"
                total_interest = 0

                while not routing.IsEnd(index_next):
                    node_index = routing.IndexToNode(index)
                    node_index_next = routing.IndexToNode(index_next)
                    route += str(node_index) + " -> "
                    # Add the distance to the next node.
                    route_dist += dist_callbacks[vehicle_nbr](node_index, node_index_next)
                    # Add the Door ID
                    route_door += str(node_to_door[node_index]) + " -> "
                    # Add the schedule
                    route_schedule += str(node_to_time[node_index])[11:-13] + " -> "
                    route_interest += str(interests[node_index]) + " -> "
                    index = index_next
                    index_next = assignment.Value(routing.NextVar(index))
                    total_interest += interests[node_index]

                node_index = routing.IndexToNode(index)
                node_index_next = routing.IndexToNode(index_next)
                route += str(node_index) + " -> " + str(node_index_next)
                route_dist += dist_callbacks[vehicle_nbr](node_index, node_index_next)
                route_door += str(node_to_door[node_index]) + " -> " + str(node_to_door[node_index_next])
                route_schedule += str(node_to_time[node_index])[11:-13] + " -> " + str(node_to_time[node_index_next])[11:-13]
                route_interest += str(interests[node_index]) + " -> " + str(interests[node_index_next])
                total_interest += interests[node_index]

                print("\nRoute for Salesman: " + str(route_to_salesman[vehicle_nbr]) + "(skill=" + str(skills[vehicle_nbr]) + ")")
                print(route)
                print(route_interest)
                print(route_door)
                print(route_schedule)
                print("Weighted Distance of Route " + str(vehicle_nbr) + ": " + str(route_dist))
                print("Total Quality serviced: " + str(total_interest))

        else:
            print('No solution found.')
    else:
        print('Specify an instance greater than 0.')


def import_data(meet_length):
    """imports data; meet_length is the time between each meeting in minutes"""
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
    # ASSUMPTION: ALL MEETINGS LAST THE SAME
    node_to_time = df_meetings["meet_time"].values
    #add Home before any meeting
    home_time = np.min(node_to_time) - np.timedelta64(meet_length, "[m]")
    node_to_time = np.concatenate(([home_time], node_to_time))
    chronometer = (node_to_time - home_time).astype("timedelta64[m]")
    chronometer = chronometer.astype("int64")

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
    #add Home interest
    interest = np.concatenate(([0], interest))

    return [location, chronometer, skill, interest, node_to_door,
            route_to_salesman, node_to_time]


if __name__ == '__main__':
    main()
