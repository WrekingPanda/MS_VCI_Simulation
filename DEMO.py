import pickle
import os
import subprocess
import traci
import numpy as np
from collections import defaultdict

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_FILE = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\sumoless_bandit_model.pkl"

# 1) UPDATE THIS to the full path of your existing net.xml
NET_FILE = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\simulation\net.net.xml"

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
DEMO_DIR = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\sumo_demo"
SIM_DURATION = 3600  # seconds (1 hour)

os.makedirs(DEMO_DIR, exist_ok=True)
print(f"[INFO] Demo directory: {DEMO_DIR}")

# --------------------------------------------------
# LOAD BANDIT MODEL
# --------------------------------------------------

print("[INFO] Loading bandit model...")
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bandit_means = model["means"]
print(f"[INFO] Loaded {len(bandit_means)} arms from model")
print("[INFO] Arms structure: (min_since_midnight, EQUIPMENTID)")

# --------------------------------------------------
# GENERATE TRIPS FROM BANDIT MODEL
# --------------------------------------------------

print("[INFO] Generating trips from bandit model...")

total_expected_volume = sum(bandit_means.values())
print(f"[INFO] Total expected volume across all arms: {total_expected_volume:.0f} vehicles")

model_time_span_sec = 24 * 60 * 60  # assume model covers 1 day
scale_factor = SIM_DURATION / model_time_span_sec
print(f"[INFO] Scaling factor: {scale_factor:.4f}")

trips = []
vehicle_id = 0

for (time_bin, equipment_id), predicted_volume in bandit_means.items():
    n_vehicles = max(1, int(predicted_volume * scale_factor))
    depart_times = np.linspace(0, SIM_DURATION - 1, n_vehicles, dtype=int)

    for depart_time in depart_times:
        # 2) UPDATE THESE edge IDs to match real edge IDs in your net.xml
        origins = ["edge_in_1", "edge_in_2"]      # <-- put real incoming edges
        destinations = ["edge_out_1", "edge_out_2"]  # <-- put real outgoing edges

        origin = np.random.choice(origins)
        destination = np.random.choice(destinations)

        trips.append({
            "id": f"veh_{vehicle_id}",
            "depart": int(depart_time),
            "from": origin,
            "to": destination,
            "arm": (time_bin, equipment_id),
        })
        vehicle_id += 1

print(f"[INFO] Generated {len(trips)} trips")

trips_file = os.path.join(DEMO_DIR, "demo.trips.xml")
with open(trips_file, "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<trips xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
    f.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/trips_file.xsd">\n')
    for trip in trips:
        f.write(
            f'    <trip id="{trip["id"]}" depart="{trip["depart"]}" '
            f'from="{trip["from"]}" to="{trip["to"]}" />\n'
        )
    f.write('</trips>\n')

print(f"[INFO] Trips file created: {trips_file}")

# --------------------------------------------------
# ROUTES VIA DUAROUTER
# --------------------------------------------------

routes_file = os.path.join(DEMO_DIR, "demo.rou.xml")
duarouter_cmd = [
    os.path.join(SUMO_HOME, "bin", "duarouter"),
    "-n", NET_FILE,
    "-t", trips_file,
    "-o", routes_file,
    "--ignore-errors",
    "--repair",
]

print("[INFO] Generating routes using duarouter...")
subprocess.run(duarouter_cmd, check=True)
print(f"[INFO] Routes created: {routes_file}")

# --------------------------------------------------
# SUMO CONFIG
# --------------------------------------------------

config_file = os.path.join(DEMO_DIR, "demo.sumocfg")
with open(config_file, "w") as f:
    f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{NET_FILE}"/>
        <route-files value="{routes_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIM_DURATION}"/>
    </time>
</configuration>
""")

print(f"[INFO] Config created: {config_file}")

# --------------------------------------------------
# RUN SUMO-GUI
# --------------------------------------------------

sumo_binary = os.path.join(SUMO_HOME, "bin", "sumo-gui")
sumo_cmd = [sumo_binary, "-c", config_file, "--start"]
print("[INFO] Starting SUMO...")
print(" ".join(sumo_cmd))
traci.start(sumo_cmd)

step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    if step % 100 == 0:
        print(f"[SIM] step {step}")
    step += 1

traci.close()
print("[INFO] Simulation finished.")
