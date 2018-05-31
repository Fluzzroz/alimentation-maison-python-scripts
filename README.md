# alimentation-maison-python-scripts
This code uses the Operational Research Tools provided by Google, see https://developers.google.com/optimization/

The goal is to find the optimal route for a number of salesmen who need to visit various appointments. This code is practice for production, but not in production mode yet.

Inputs:
- a list of locations (door_id, longitude/latitude, human readable address)
- a list of meetings (door_id, meeting time, interest)
- a list of salesmen (salesman_id, skill, name)
Currently the inputs are csv files generated by the database. In production, it will be a direct link to the database.

Outputs: 
- None. The script will print in the console the routes for all salesman. In a production, it will be a json export.


The main file "tsp.py" will attribute a route for each salesman that will respect the following constraints:
- all meetings must be attended to
- we assume that the meeting length + travel time is always the same (1 hour), thus this is how we spaced the meeting schedule
- we optimize for the closest *weighted* distance covered over all the salesmen
- the base distance is calculated by using the longitude and latitude coordinates between two locations and assuming the earth is a perfect sphere. It is not the real-world street route between two addresses, but rather a direct "flying" distance. Still, the approximation works well.
- the weights ensure that good meetings are favored to good salesmen and poor meetings are favored to poor salesman. They are proportional to the difference between the meeting quality (interest) and the salesman skill. For example, a low quality meeting (interest = 1) will create a big penalty to a good salesman (skill 10) and a small penalty to a poor salesman (skill 2). Thus, if a good and a poor salesmen are both at a similar distance to the same low interest meeting, the poor salesman will get it. Vice-versa for high-quality meetings.
- we assume the starting point of each route is the location of the first meeting. Since the or-tools library was coded with a depot in mind, this is done by adding a "depot" location meeting 1 hour before the true first meeting and assuming that the distance between any node and the depot is 0. Therefore, the only consideration for the first meeting location is that it optimizes the route as a whole.

The locations are called "doors" by the client and "nodes" by the or-tools library.




