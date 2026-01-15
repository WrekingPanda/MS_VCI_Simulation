import os
import sys
import random
from collections import defaultdict
import traci
import pandas as pd
import numpy as np
import time
import pickle
import sumolib
import random
import xml.etree.ElementTree as ET

# Define your local SUMO bin path (Check if this matches your actual installation path)
sumo_home = "C:/Program Files (x86)/Eclipse/Sumo"

tools_path = os.path.join(sumo_home, 'tools')

if tools_path not in sys.path:
    sys.path.append(tools_path)

# Load Data
# Change this to your sumo.exe path, for faster training without the gui
SUMO_BINARY = ".../sumo.exe"  # or "sumo-gui" for visualization
SUMO_CFG = "vci.sumocfg"

# Possible origins
START_EDGES_C = ["1019723","1019718","1063513","1062943","1020382","2842002","1020167","1020160","2842000","1516937","1016643","1020050","1020045","1019568","1210462","1582715","1405109","1302643","1038024","1215193","2016241","1810069","1204005","1401473","1204022","1189928"]
START_EDGES_D = ["1189910","1401479","1175990","1111269.111","1111267","1215241","1181539","1111242","1216119","1122615","1016648","1188031","1255432","1020165","1768757","1949246","1306152","1062246","1122691","1019722","1019716","1401214"]
# Possible destinations
END_EDGES_C = ["1019719","1122692","1063262","1047385","1076349","1016658","1020171","1020157","1888721","1888720","2020471","1016644","1020046","1020029","1019569","1210226","1111488","1036157","1016681","1215220","1935065","1175980","1401472","1189936","1210069"]
END_EDGES_D = ["1401537","1189937","1203952","1810067","1111269","1188014","1215352","1181481","1111240","1214240","1019567","1020004","2006681","1019667","1020161","1020174","1016657","1054906","1047469","1062317","1063338","1019717","1051044","1972190"]

# Load preprocessed data
train_df = pd.read_csv("./Dataset/simple_train.csv", parse_dates=["AGG_PERIOD_START"])
test_df = pd.read_csv("./Dataset/simple_test.csv", parse_dates=["AGG_PERIOD_START"])

# Ensure that the arm keys are tuples
def ensure_half_arm(df):
    if isinstance(df["half_arm"].iloc[0], tuple):
        return df

    def parse_arm(s):
        return eval(s)
    df = df.copy()
    df["half_arm"] = df["half_arm"].apply(parse_arm)
    return df

train_df = ensure_half_arm(train_df)
test_df = ensure_half_arm(test_df)

def add_date(df):
    df = df.copy()
    df["date"] = df["AGG_PERIOD_START"].dt.date
    return df

def get_real_counts_for_day(df_day):
    g = df_day.groupby("half_arm")["TOTAL_VOLUME"].sum()
    return g.to_dict()

# Map the real sensors to the SUMO induction loop detectors introduced by us
sensor_id_to_detectors = {121726: ["121726_0", "121726_1", "121726_2"],
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
                          121756: ["121756_0", "121756_1"]}

# Map the real sensors to the direction of the road they monitor ('C' or 'D')
sensor_direction = {121726: 'D',
                    121727: 'D',
                    121731: 'C',
                    121732: 'C',
                    121733: 'C',
                    121734: 'C',
                    121735: 'C',
                    121736: 'C',
                    121741: 'C',
                    121742: 'C',
                    121754: 'D',
                    121755: 'C',
                    121756: 'C'}

# Map the sensors to the road segment they're on
sensor_to_edge = {"121726": "1254022",
                  "121727": "1214665",
                  "121731": "1063682",
                  "121732": "1062244",
                  "121733": "1062753",
                  "121734": "1051025",
                  "121735": "1051026",
                  "121736": "1062939",
                  "121741": "1254019",
                  "121742": "1254017",
                  "121754": "1175979",
                  "121755": "1176005",
                  "121756": "1181559"}

# Read the SUMO output file and aggregate flow counts.
def parse_sumo_detector_output(xml_file, det_lookup):
    sim_counts = defaultdict(int)
    
    # Parse the XML file
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for interval in root.findall('interval'):
            det_id = interval.get('id')
            
            # Only process if this is one of our known sensors
            if det_id in det_lookup:
                # Get the count for this interval
                count = int(interval.get('nVehContrib', 0))
                
                if count > 0:
                    sensor_id, direction = det_lookup[det_id]
                    
                    # Calculate the time bin
                    # 'begin' is in seconds (e.g., 0.00, 1800.00)
                    begin_time = float(interval.get('begin'))
                    half_bin = int(begin_time // 1800)
                    
                    # Add to our aggregate
                    arm_key = (half_bin, sensor_id, direction)
                    sim_counts[arm_key] += count
                    
    except FileNotFoundError:
        print(f"[ERROR] Could not find {xml_file}. Did SUMO crash?")
        
    return dict(sim_counts)

# Load the map network
net = sumolib.net.readNet("net.net.xml")

def get_connected_edges(sensor_id, depth=3, direction="upstream"):
    """
    Finds N edges either 'behind' (upstream) or 'after' (downstream) the sensor.
    """
    sensor_edge_id = sensor_to_edge[sensor_id]
    sensor_edge = net.getEdge(sensor_edge_id)
    found_edges = []
    queue = [sensor_edge]
    visited = {sensor_edge_id}
    
    while queue and len(found_edges) < depth:
        current_edge = queue.pop(0)
        
        # Determine which neighbors to look at
        if direction == "upstream":
            neighbors = current_edge.getFromNode().getIncoming()
        else: # downstream
            neighbors = current_edge.getToNode().getOutgoing()
            
        for n_edge in neighbors:
            n_id = n_edge.getID()
            if n_id not in visited:
                found_edges.append(n_id)
                visited.add(n_id)
                queue.append(n_edge)
    return found_edges if found_edges else [sensor_edge_id]

def build_sensor_edge_mapping():
    mapping = {}
    sensor_direction = {121726: 'D', 121727: 'D', 121731: 'C', 121732: 'C', 121733: 'C', 121734: 'C', 121735: 'C', 121736: 'C', 121741: 'C', 121742: 'C', 121754: 'D', 121755: 'C', 121756: 'C'}

    for s_id, direction in sensor_direction.items():
        sensor_id = str(s_id)
        edge_id = sensor_to_edge[sensor_id]
        
        # Get 3 edges before the sensor
        possible_starts = get_connected_edges(sensor_id, depth=3, direction="upstream")
        
        # Get 7 edges after the sensor
        possible_ends = get_connected_edges(sensor_id, depth=7, direction="downstream")
        
        mapping[(s_id, direction)] = (possible_starts, edge_id, possible_ends)
    return mapping

# Simple independent Gaussian Thompson Sampling per arm.
# Reward = -loss; here we will approximate per-arm reward using negative squared error between sim and real volume.
class GaussianThompsonBandit:
    def __init__(self, arms, init_mean=20.0, init_var=100.0, obs_noise_var=100.0):
        self.arms = list(arms)
        self.obs_noise_var = obs_noise_var
        self.means = {a: init_mean for a in self.arms}
        self.vars  = {a: init_var  for a in self.arms}

    def sample_flows(self):
        flows = {}
        MAX_FLOW = 7000.0
        for a in self.arms:
            mu = self.means[a]
            var = self.vars[a]

            # Sample from the normal distribution
            sample = np.random.normal(mu, np.sqrt(var))

            # Clip the value to ensure it stays within physically possible limits
            flows[a] = np.clip(sample, 0.0, MAX_FLOW)
        return flows

    def update(self, real_counts, sim_counts, reward=None):
        # Pick one arm to monitor (Optional)
        debug_arm = next(iter(self.arms)) if self.arms else None
        before = (self.means.get(debug_arm), self.vars.get(debug_arm)) if debug_arm else None

        updated_count = 0
        for a, y in real_counts.items():
            if a in self.means:
                mu_prior = self.means[a]
                var_prior = self.vars[a]
                var_noise = self.obs_noise_var

                # Bayesian update for normal-normal model
                var_post = 1.0 / (1.0 / var_prior + 1.0 / var_noise)
                mu_post = var_post * (mu_prior / var_prior + y / var_noise)

                self.means[a] = mu_post
                self.vars[a]  = var_post
                updated_count += 1
            else:
                # This will trigger if the arm keys don't match
                print(f"[DEBUG] Key mismatch: {a} not found in bandit arms!")
        print(f"[BANDIT] Successfully updated {updated_count} out of {len(real_counts)} real counts.")
        
    def sample_flows_subset(self, arms_subset):
        MAX_FLOW = 7000.0
        flows = {}
        for a in arms_subset:
            mu = self.means[a]
            var = self.vars[a]
            flows[a] = np.clip(
                np.random.normal(mu, np.sqrt(var)), 0.0, MAX_FLOW
            )
        return flows

    def save(self, filepath):
        data = {
            'means': self.means,
            'vars': self.vars,
            'arms': self.arms,
            'obs_noise_var': self.obs_noise_var
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct the bandit with saved state
        bandit = GaussianThompsonBandit(data['arms'], obs_noise_var=data['obs_noise_var'])
        bandit.means = data['means']
        bandit.vars = data['vars']
        print(f"[INFO] Model loaded from {filepath}")
        return bandit
    
def build_trips_for_day(flow_estimates, sensor_edge_map, time_bins_per_arm):
    trips = []
    v_idx = 0
    for arm, flow_val in flow_estimates.items():
        half_bin, sensor_id, direction = arm
        FLOW_SCALE = 0.1  # simulate 10%

        n = int(round(flow_val * FLOW_SCALE))
        
        if n == 0 or (sensor_id, direction) not in sensor_edge_map:
            continue

        possible_starts, sensor_edge, possible_ends = sensor_edge_map[(sensor_id, direction)]
        t0, t1 = time_bins_per_arm[arm]

        for _ in range(n):
            # Randomly pick from the verified local edges
            start_node = random.choice(possible_starts)
            end_node = random.choice(possible_ends)
            
            depart = random.uniform(t0, t1)
            veh_id = f"veh_{v_idx}"
            
            # (veh_id, depart, from, via, to)
            trips.append((veh_id, depart, start_node, sensor_edge, end_node))
            v_idx += 1
    return trips

def build_time_bins_for_half_hour():
    mapping = {}
    for hb in range(48):
        start_min = hb * 30
        end_min   = start_min + 30
        mapping[hb] = (start_min * 60, end_min * 60)
    return mapping

def build_time_bins_per_arm(arms):
    half_bin_map = build_time_bins_for_half_hour()
    return {a: half_bin_map[a[0]] for a in arms}

def write_trips_xml(trips, filename="daily_trips.rou.xml"):
    # Trips need to be ordered by departure time
    trips.sort(key=lambda x: x[1])
    with open(filename, "w") as f:
        f.write('<routes>\n')
        # Define a standard vehicle type
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="70"/>\n')
        
        for veh_id, depart, start_edge, end_edge, arm in trips:
            # Write trip directly to XML
            f.write(f'    <vehicle id="{veh_id}" type="car" depart="{depart}" departLane="best" departSpeed="max">\n')
            f.write(f'        <route edges="{start_edge} {end_edge}"/>\n')
            f.write('    </vehicle>\n')
            
        f.write('</routes>')

def run_sumo_episode(trips, sensor_edge_map, sensor_id_to_detectors):
    print(f"[SUMO] Starting episode with {len(trips)} trips.")
    
    # Generate Route File
    route_filename = "daily_trips.rou.xml"
    write_trips_xml(trips, route_filename)

    # Configure SUMO
    sumo_cmd = [
        SUMO_BINARY, 
        "-c", SUMO_CFG,
        "-r", route_filename,
        "--mesosim", "true",
        "--meso-overtaking", "false",
        "--step-length", "2",
        "--no-step-log", "true",
        "--log", "sumo_error.log",        # Check this file if it crashes!
        "--ignore-route-errors", "true",
        "--quit-on-end", "true"
    ]

    # Start TraCI (Automatic Port Selection)
    traci.start(sumo_cmd)
    # Wait briefly for connection to stabilize
    time.sleep(0.5)

    sim_counts = defaultdict(int)

    try:
        # Subscribe to Detectors
        unique_sensors = {arm[1] for _, _, _, _, arm in trips}
        for sensor_id in unique_sensors:
            det_ids = sensor_id_to_detectors.get(sensor_id, [])
            for det_id in det_ids:
                # 0x10 is the constant for LAST_STEP_VEHICLE_NUMBER
                traci.inductionloop.subscribe(det_id, [0x10])
        
        # Simulation Loop
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Retrieve subscription results efficiently
            for sensor_id in unique_sensors:
                det_ids = sensor_id_to_detectors.get(sensor_id, [])
                for det_id in det_ids:
                    res = traci.inductionloop.getSubscriptionResults(det_id)
                    if res and (0x10 in res):
                        count = res[0x10]
                        if count > 0:
                            pass 
        
    except Exception as e:
        print(f"[SUMO ERROR] {e}")
        # If it crashes, print the log file content
        if os.path.exists("sumo_error.log"):
            with open("sumo_error.log", "r") as f:
                print("--- SUMO LOG START ---")
                print(f.read())
                print("--- SUMO LOG END ---")
        raise e

    finally:
        traci.close()
        time.sleep(1.0) # Give OS time to release the port
        print("[SUMO] Connection closed.")

    # Parse Results from File
    print("[SUMO] Reading results from XML...")
    det_lookup = {}
    for s_id, det_list in sensor_id_to_detectors.items():
        d = sensor_direction.get(s_id)
        for det_id in det_list:
            det_lookup[det_id] = (s_id, d)
            
    sim_counts = parse_sumo_detector_output("detectors.out.xml", det_lookup)
    
    return sim_counts

def compute_loss(sim_counts, real_counts, loss_type="l1"):
    loss = 0.0

    # Union of arms present in either dict
    all_arms = set(real_counts.keys()) | set(sim_counts.keys())

    for arm in all_arms:
        y_real = real_counts.get(arm, 0.0)
        y_sim  = sim_counts.get(arm, 0.0)
        diff = y_sim - y_real

        if loss_type == "l2":
            loss += diff ** 2
        else:  # "l1"
            loss += abs(diff)

    return loss

def train_bandit_with_sumo(train_df, num_episodes=None):
    # Ensure half_arm is a tuple
    train_df = ensure_half_arm(train_df)
    train_df = add_date(train_df)

    # Arms from training data
    arms = sorted(train_df["half_arm"].unique())
    bandit = GaussianThompsonBandit(arms)

    # Mapping from (sensor, dir) to (start_edge, end_edge)
    sensor_edge_map = build_sensor_edge_mapping()
    
    # Maps the real sensors to the SUMO induction loop detectors introduced by us
    sensor_id_to_detectors = {121726: ["121726_0", "121726_1", "121726_2"],
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
                            121756: ["121756_0", "121756_1"]}

    # Time-bin mapping
    time_bins_per_arm = build_time_bins_per_arm(arms)

    # Group by day
    grouped = train_df.groupby("date")
    days = list(grouped.groups.keys())
    if num_episodes is not None:
        days = days[:num_episodes]

    print(f"[INFO] Training on {len(days)} days, {len(arms)} arms.")

    for day_idx, day in enumerate(days, start=1):
        df_day = grouped.get_group(day)
        print(f"\n[DAY {day_idx}/{len(days)}] {day}")

        # Real counts per arm for this day
        real_counts = get_real_counts_for_day(df_day)
        print(f"[DAY {day}] unique arms today: {len(real_counts)}")

        # Bandit samples flow estimates per arm
        active_arms = real_counts.keys()
        flow_estimates = bandit.sample_flows_subset(active_arms)
        print(f"[DAY {day}] sampled flows (first 5): {list(flow_estimates.items())[:5]}")

        # Build trips for this day from flow estimates
        trips = build_trips_for_day(flow_estimates, sensor_edge_map, time_bins_per_arm)
        print(f"[DAY {day}] generated {len(trips)} trips.")

        # Run SUMO and get simulated counts per arm
        sim_counts = run_sumo_episode(trips, sensor_edge_map, sensor_id_to_detectors)
        
        # Compute scaled loss for logging and reward
        # Instead of a raw sum, use the average error per arm (MAE)
        raw_loss = compute_loss(sim_counts, real_counts, loss_type="l1")

        # Scaling Factor: Normalize by the number of arms tracked today
        num_arms_today = len(real_counts)
        if num_arms_today > 0:
            mae = raw_loss / num_arms_today
        else:
            mae = 0

        # Reward Transformation (Using a scaling factor keeps the reward in a smaller range, which prevents the Bandit's variance from collapsing or exploding)
        reward = -(mae * 0.1)
        print(f"[DAY {day}] Total Loss: {raw_loss:.2f} | MAE: {mae:.2f} | Scaled Reward: {reward:.2f}")

        # Update bandit with real and simulated counts
        bandit.update(real_counts, sim_counts, reward)
        print(f"[DAY {day}] bandit updated.")

        # Temporary save each 10 days
        if day_idx%10 == 0:
            bandit.save("vci_bandit_model.pkl")

    return bandit

# Train the model
print(f"Start Training")
trained_bandit = train_bandit_with_sumo(train_df, 730)

# Save the final model
trained_bandit.save("vci_bandit_model.pkl")