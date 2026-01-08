import os
import random
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import traci
import traci.constants as tc
import sumolib

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
SUMO_CFG = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\simulation\vci.sumocfg"
NET_FILE = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\simulation\net.net.xml"

TEST_CSV = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\Dataset\simple_test.csv"
MODEL_FILE = "vci_bandit_model.pkl"

ROUTE_FILE = "temp_test_trips.rou.xml"

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

df = pd.read_csv(TEST_CSV, parse_dates=["AGG_PERIOD_START"])

def ensure_half_arm(df):
    if isinstance(df["half_arm"].iloc[0], tuple):
        return df
    df = df.copy()
    df["half_arm"] = df["half_arm"].apply(eval)
    return df

df = ensure_half_arm(df)
df["date"] = df["AGG_PERIOD_START"].dt.date

# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bandit_means = model["means"]

# ----------------------------------------------------------------------
# NETWORK
# ----------------------------------------------------------------------

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

sensor_id_to_detectors = {
    121726: ["121726_0", "121726_1", "121726_2"],
    121727: ["121727_0", "121727_1", "121727_2"],
    121731: ["121731_0", "121731_1", "121731_2", "121731_3"],
    121732: ["121732_0", "121732_1", "121732_2", "121732_3"],
    121733: ["121733_0", "121733_1", "121733_2"],
    121734: ["121734_0", "121734_1", "121734_2", "121734_3"],
    121735: ["121735_0", "121735_1", "121735_2"],
    121736: ["121736_0", "121736_1", "121736_2"],
    121741: ["121741_0", "121741_1", "121741_2"],
    121742: ["121742_0", "121742_1", "121742_2"],
    121754: ["121754_0", "121754_1"],
    121755: ["121755_0", "121755_1"],
    121756: ["121756_0", "121756_1"]
}

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

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

def write_trips(trips):
    with open(ROUTE_FILE, "w") as f:
        f.write("<routes>\n")
        f.write('<vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="70"/>\n')
        for vid, dep, fr, to in trips:
            f.write(
                f'<trip id="{vid}" depart="{dep:.2f}" from="{fr}" to="{to}" type="car"/>\n'
            )
        f.write("</routes>\n")

# ----------------------------------------------------------------------
# SIMULATION TEST
# ----------------------------------------------------------------------

metrics = []

for day, df_day in df.groupby("date"):

    print(f"\n[TEST DAY] {day}")

    trips = []
    vid = 0

    for arm in df_day["half_arm"].unique():
        hb, sensor_id, d = arm
        mean_flow = bandit_means.get(arm, 0.0)

        n = np.random.poisson(mean_flow)
        if n == 0 or (sensor_id, d) not in sensor_edge_map:
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

    write_trips(trips)

    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "-r", ROUTE_FILE, "--no-step-log", "true"])

    sim_counts = defaultdict(int)

    for sid, dets in sensor_id_to_detectors.items():
        for d in dets:
            traci.inductionloop.subscribe(d, [tc.LAST_STEP_VEHICLE_NUMBER])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        hb = int(sim_time // 1800)

        for sid, dets in sensor_id_to_detectors.items():
            c = sum(
                traci.inductionloop.getSubscriptionResults(d).get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
                for d in dets
            )
            if c > 0:
                sim_counts[(hb, sid, sensor_direction[sid])] += c

    traci.close()

    real_counts = df_day.groupby("half_arm")["TOTAL_VOLUME"].sum().to_dict()

    for arm in set(real_counts) | set(sim_counts):
        r = real_counts.get(arm, 0)
        s = sim_counts.get(arm, 0)

        metrics.append({
            "date": day,
            "arm": arm,
            "real": r,
            "sim": s,
            "abs_err": abs(s - r),
            "sq_err": (s - r) ** 2,
            "ape": abs(s - r) / max(r, 1)
        })

# ----------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------

dfm = pd.DataFrame(metrics)

print("\n================ SUMO POLICY TEST RESULTS ================")
print(f"MAE  : {dfm.abs_err.mean():.2f}")
print(f"RMSE : {np.sqrt(dfm.sq_err.mean()):.2f}")
print(f"MAPE : {dfm.ape.mean() * 100:.2f}%")
print("=========================================================")
