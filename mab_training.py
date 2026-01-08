# --------------------------------------------------
# IMPORTS + LOAD MODEL
# --------------------------------------------------

import os
import pickle
import subprocess
import numpy as np

from mab_training import START_EDGES_C, START_EDGES_D, END_EDGES_C, END_EDGES_D  # <— reuse your lists

MODEL_FILE = r"...\sumoless_bandit_model.pkl"
NET_FILE   = r"...\simulation\net.net.xml"
SUMO_HOME  = r"C:\Program Files (x86)\Eclipse\Sumo"
DEMO_DIR   = r"...\sumo_demo"
SIM_DURATION = 3600  # 1h

os.makedirs(DEMO_DIR, exist_ok=True)

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bandit_means = model["means"]

# --------------------------------------------------
# TRIPS FROM BANDIT (UPDATED EDGES)
# --------------------------------------------------

trips = []
vehicle_id = 0
model_time_span_sec = 24 * 60 * 60
scale_factor = SIM_DURATION / model_time_span_sec

for (time_bin, equipment_id), predicted_volume in bandit_means.items():
    n_vehicles = max(1, int(predicted_volume * scale_factor))
    depart_times = np.linspace(0, SIM_DURATION - 1, n_vehicles, dtype=int)

    # Simple heuristic: choose direction based on lane bundle, if you have it.
    # For now, random C/D split to re‑use your directional edge sets:
    if np.random.rand() < 0.5:
        origins      = START_EDGES_C
        destinations = END_EDGES_C
    else:
        origins      = START_EDGES_D
        destinations = END_EDGES_D

    for depart in depart_times:
        trips.append({
            "id": f"veh_{vehicle_id}",
            "depart": int(depart),
            "from": np.random.choice(origins),
            "to":   np.random.choice(destinations),
        })
        vehicle_id += 1

# --------------------------------------------------
# WRITE demo.trips.xml (same as before, but correct structure)
# --------------------------------------------------

trips_file = os.path.join(DEMO_DIR, "demo.trips.xml")
with open(trips_file, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<trips xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
    f.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/trips_file.xsd">\n')
    for trip in trips:
        f.write(
            f'    <trip id="{trip["id"]}" depart="{trip["depart"]}" '
            f'from="{trip["from"]}" to="{trip["to"]}"/>\n'
        )
    f.write('</trips>\n')

# --------------------------------------------------
# duarouter CALL (use --route-files)
# --------------------------------------------------

routes_file = os.path.join(DEMO_DIR, "demo.rou.xml")
duarouter = os.path.join(SUMO_HOME, "bin", "duarouter")

cmd = [
    duarouter,
    "-n", NET_FILE,
    "--route-files", trips_file,  # <— fix deprecation
    "-o", routes_file,
    "--ignore-errors",
    "--repair",
]

subprocess.run(cmd, check=True)
