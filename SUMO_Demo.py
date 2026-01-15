import random
import pickle
import numpy as np
import pandas as pd
import sumolib
import traci

# CONFIG
# Set the paths for your machine (Check if this matches your actual installation path)
SUMO_GUI = "C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

SUMO_CFG = "vci.sumocfg"
NET_FILE = "net.net.xml"
MODEL_FILE = "vci_bandit_model.pkl"
ROUTE_FILE = "demo_bandit_routes.rou.xml"

# LOAD MODEL
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bandit_means = model["means"]
arms = model["arms"]

print(f"[INFO] Loaded bandit model with {len(arms)} arms.")

# NETWORK
net = sumolib.net.readNet(NET_FILE)

sensor_to_edge = {
    "121726": "1254022", "121727": "1214665", "121731": "1063682",
    "121732": "1062244", "121733": "1062753", "121734": "1051025",
    "121735": "1051026", "121736": "1062939", "121741": "1254019",
    "121742": "1254017", "121754": "1175979", "121755": "1176005",
    "121756": "1181559"
}

sensor_direction = {
    121726: 'D', 121727: 'D', 121731: 'C', 121732: 'C',
    121733: 'C', 121734: 'C', 121735: 'C', 121736: 'C',
    121741: 'C', 121742: 'C', 121754: 'D', 121755: 'C', 121756: 'C'
}

# ROUTE RESTRICTIONS (IDENTICAL TO TRAINING)
def get_connected_edges(sensor_id, depth, direction):
    edge = net.getEdge(sensor_to_edge[sensor_id])
    found = [edge.getID()]
    queue = [edge]
    visited = set(found)

    while queue and len(found) < depth:
        e = queue.pop(0)
        neighbors = (
            e.getFromNode().getIncoming()
            if direction == "upstream"
            else e.getToNode().getOutgoing()
        )
        for n in neighbors:
            if n.getID() not in visited:
                visited.add(n.getID())
                found.append(n.getID())
                queue.append(n)
    return found

sensor_edge_map = {
    (sid, d): (
        get_connected_edges(str(sid), 3, "upstream"),
        get_connected_edges(str(sid), 7, "downstream")
    )
    for sid, d in sensor_direction.items()
}

# BUILD DEMO TRAFFIC
trips = []
vid = 0

for arm, mean_flow in bandit_means.items():
    hb, sensor_id, d = arm

    if (sensor_id, d) not in sensor_edge_map:
        continue

    n = np.random.poisson(mean_flow)
    if n == 0:
        continue

    starts, ends = sensor_edge_map[(sensor_id, d)]
    t0, t1 = hb * 1800, (hb + 1) * 1800

    for _ in range(n):
        trips.append((
            f"veh_{vid}",
            random.uniform(t0, t1),
            random.choice(starts),
            random.choice(ends)
        ))
        vid += 1

print(f"[INFO] Generated {len(trips)} vehicles for visual demo.")

# WRITE ROUTES
trips.sort(key=lambda x: x[1])

with open(ROUTE_FILE, "w") as f:
    f.write("<routes>\n")
    f.write('<vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="70"/>\n')

    for vid, dep, fr, to in trips:
        f.write(
            f'<trip id="{vid}" depart="{dep:.2f}" from="{fr}" to="{to}" type="car"/>\n'
        )

    f.write("</routes>\n")

print(f"[INFO] Route file written: {ROUTE_FILE}")

# START SUMO GUI
traci.start([
    SUMO_GUI,
    "-c", SUMO_CFG,
    "-r", ROUTE_FILE,
    "--start",
    "--quit-on-end"
])

print("[INFO] SUMO-GUI started")

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

traci.close()
print("[INFO] Demo finished.")
