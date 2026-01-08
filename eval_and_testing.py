import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import traci
import traci.constants as tc
import sumolib
import pickle
import time

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
SUMO_CFG = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\simulation\vci.sumocfg"
NET_FILE = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\simulation\net.net.xml"

TEST_CSV = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\Dataset\simple_test.csv"
MODEL_FILE = "vci_bandit_model.pkl"

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

print("[INFO] Loading test dataset...")
test_df = pd.read_csv(TEST_CSV, parse_dates=["AGG_PERIOD_START"])

def ensure_half_arm(df):
    if isinstance(df["half_arm"].iloc[0], tuple):
        return df
    df = df.copy()
    df["half_arm"] = df["half_arm"].apply(eval)
    return df

def add_date(df):
    df = df.copy()
    df["date"] = df["AGG_PERIOD_START"].dt.date
    return df

test_df = add_date(ensure_half_arm(test_df))

# ----------------------------------------------------------------------
# NETWORK & SENSOR MAPS
# ----------------------------------------------------------------------

print("[INFO] Loading SUMO network...")
net = sumolib.net.readNet(NET_FILE)

sensor_to_edge = {
    "121726": "1254022", "121727": "1214665", "121731": "1063682",
    "121732": "1062244", "121733": "1062753", "121734": "1051025",
    "121735": "1051026", "121736": "1062939", "121741": "1254019",
    "121742": "1254017", "121754": "1175979", "121755": "1176005",
    "121756": "1181559"
}

sensor_direction = {
    121726: 'D', 121727: 'D', 121731: 'C', 121732: 'C', 121733: 'C',
    121734: 'C', 121735: 'C', 121736: 'C', 121741: 'C', 121742: 'C',
    121754: 'D', 121755: 'C', 121756: 'C'
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
# UTILITIES
# ----------------------------------------------------------------------

def get_real_counts_for_day(df_day):
    return df_day.groupby("half_arm")["TOTAL_VOLUME"].sum().to_dict()

def build_time_bins_for_half_hour():
    return {hb: (hb * 1800, (hb + 1) * 1800) for hb in range(48)}

def build_time_bins_per_arm(arms):
    half_map = build_time_bins_for_half_hour()
    return {a: half_map[a[0]] for a in arms}

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

def build_sensor_edge_mapping():
    mapping = {}
    for sid, d in sensor_direction.items():
        sid_str = str(sid)
        mapping[(sid, d)] = (
            get_connected_edges(sid_str, 3, "upstream"),
            sensor_to_edge[sid_str],
            get_connected_edges(sid_str, 7, "downstream")
        )
    return mapping

def write_trips_xml(trips, filename):
    trips.sort(key=lambda x: x[1])
    with open(filename, "w") as f:
        f.write("<routes>\n")
        f.write('<vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="70"/>\n')
        for vid, dep, fr, to in trips:
            f.write(
                f'<trip id="{vid}" depart="{dep:.2f}" from="{fr}" to="{to}" type="car"/>\n'
            )
        f.write("</routes>\n")

# ----------------------------------------------------------------------
# LOAD TRAINED MODEL
# ----------------------------------------------------------------------

print("[INFO] Loading trained bandit model...")
with open(MODEL_FILE, "rb") as f:
    data = pickle.load(f)

bandit_means = data["means"]
arms = data["arms"]

print(f"[INFO] Loaded model with {len(arms)} arms.")

# ----------------------------------------------------------------------
# TEST LOOP
# ----------------------------------------------------------------------

sensor_edge_map = build_sensor_edge_mapping()
time_bins = build_time_bins_per_arm(arms)

metrics = []

grouped = test_df.groupby("date")
days = list(grouped.groups.keys())
total_days = len(days)

print(f"\n[INFO] Starting evaluation on {total_days} test days.")

global_start = time.time()

for day_idx, day in enumerate(days, start=1):
    day_start = time.time()
    df_day = grouped.get_group(day)

    print(f"\n[TEST DAY {day_idx}/{total_days}] {day}")

    real_counts = get_real_counts_for_day(df_day)

    # ---------------- Build Trips ----------------
    trips = []
    vid = 0
    for arm, mean_flow in bandit_means.items():
        hb, sensor_id, d = arm
        n = int(round(mean_flow))
        if n == 0 or (sensor_id, d) not in sensor_edge_map:
            continue

        starts, _, ends = sensor_edge_map[(sensor_id, d)]
        t0, t1 = time_bins[arm]

        for _ in range(n):
            trips.append((
                f"veh_{vid}",
                random.uniform(t0, t1),
                random.choice(starts),
                random.choice(ends)
            ))
            vid += 1

    print(f"    [INFO] Generated {len(trips)} vehicles.")

    # ---------------- Run SUMO ----------------
    route_file = "temp_test_trips.rou.xml"
    write_trips_xml(trips, route_file)

    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "-r", route_file,
        "--no-step-log", "true"
    ])

    sim_counts = defaultdict(int)

    for s_id, dets in sensor_id_to_detectors.items():
        for d in dets:
            traci.inductionloop.subscribe(d, [tc.LAST_STEP_VEHICLE_NUMBER])

    sim_end_time = 24 * 3600
    last_print = -1

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        hb = int(sim_time // 1800)

        progress = int((sim_time / sim_end_time) * 10)
        if progress != last_print:
            print(f"    [SUMO] Progress: {progress * 10}%")
            last_print = progress

        for s_id, dets in sensor_id_to_detectors.items():
            c = sum(
                traci.inductionloop
                .getSubscriptionResults(d)
                .get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
                for d in dets
            )
            if c > 0:
                sim_counts[(hb, s_id, sensor_direction[s_id])] += c

    traci.close()

    print(f"    [INFO] SUMO finished.")
    print(f"    [DAY SUMMARY] Real arms: {len(real_counts)} | Simulated arms: {len(sim_counts)}")

    # ---------------- Metrics ----------------
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

    print(f"    [INFO] Day runtime: {time.time() - day_start:.1f}s")

# ----------------------------------------------------------------------
# FINAL RESULTS
# ----------------------------------------------------------------------

dfm = pd.DataFrame(metrics)

print("\n================ FINAL TEST RESULTS ================")
print(f"MAE  : {dfm.abs_err.mean():.2f}")
print(f"RMSE : {np.sqrt(dfm.sq_err.mean()):.2f}")
print(f"MAPE : {dfm.ape.mean() * 100:.2f}%")
print("===================================================")

print(f"[INFO] Total evaluation time: {time.time() - global_start:.1f}s")
