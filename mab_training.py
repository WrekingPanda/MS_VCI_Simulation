import os
import sys
import random
from collections import defaultdict
import traci
import pandas as pd
import numpy as np
import io
import time

# ----------------------------------------------------------------------
# LOAD DATA & CONFIG
# ----------------------------------------------------------------------

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"          # or "sumo-gui"
SUMO_CFG = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\simulation\vci.sumocfg"  # must reference your .net.xml

# Possible origins
START_EDGES_C = ["1019723","1019718","1063513","1062943","1020382","2842002","1020167","1020160","2842000","1516937","1016643","1020050","1020045","1019568","1210462","1582715","1405109","1302643","1038024","1215193","2016241","1810069","1204005","1401473","1204022","1189928"]
START_EDGES_D = ["1189910","1401479","1175990","1111269.111","1111267","1215241","1181539","1111242","1216119","1122615","1016648","1188031","1255432","1020165","1768757","1949246","1306152","1062246","1122691","1019722","1019716","1401214"]
# Possible destinations
END_EDGES_C = ["1019719","1122692","1063262","1047385","1076349","1016658","1020171","1020157","1888721","1888720","2020471","1016644","1020046","1020029","1019569","1210226","1111488","1036157","1016681","1215220","1935065","1175980","1401472","1189936","1210069"]
END_EDGES_D = ["1401537","1189937","1203952","1810067","1111269","1188014","1215352","1181481","1111240","1214240","1019567","1020004","2006681","1019667","1020161","1020174","1016657","1054906","1047469","1062317","1063338","1019717","1051044","1972190"]

# Load preprocessed data
train_df = pd.read_csv(r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\Dataset\simple_train.csv", parse_dates=["AGG_PERIOD_START"])
test_df = pd.read_csv(r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\Dataset\simple_test.csv", parse_dates=["AGG_PERIOD_START"])

def ensure_half_arm(df):
    """
    Ensure df['half_arm'] is a tuple: (half_time_bin, EQUIPMENTID, LANE_BUNDLE_DIRECTION)
    """
    if isinstance(df["half_arm"].iloc[0], tuple):
        return df

    # Example if it's a string like "(bin, sensor, 'C')": parse it
    def parse_arm(s):
        # Adjust this parser depending on your actual format
        # Example: "(10, 121726, 'C')" -> (10, 121726, 'C')
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
    """
    Aggregate TOTAL_VOLUME per arm for one day.
    Returns {arm: volume}.
    """
    g = df_day.groupby("half_arm")["TOTAL_VOLUME"].sum()
    return g.to_dict()

def build_sensor_edge_mapping():
    """
    Build a mapping: (sensor_id, dir) -> (start_edge, end_edge).

    TODO: Replace this heuristic with your actual mapping, based on
          how each AEDL sensor corresponds to SUMO edges and direction.
    """
    mapping = {}
    # Example: round-robin assignment for demo
    # WARNING: replace with real mapping!
    for i, s in enumerate(sorted(train_df["EQUIPMENTID"].unique())):
        start_c = START_EDGES_C[i % len(START_EDGES_C)]
        end_c   = END_EDGES_C[i % len(END_EDGES_C)]
        mapping[(s, "C")] = (start_c, end_c)

    for i, s in enumerate(sorted(train_df["EQUIPMENTID"].unique())):
        start_d = START_EDGES_D[i % len(START_EDGES_D)]
        end_d   = END_EDGES_D[i % len(END_EDGES_D)]
        mapping[(s, "D")] = (start_d, end_d)

    return mapping

class GaussianThompsonBandit:
    """
    Simple independent Gaussian Thompson Sampling per arm.
    Reward = -loss; here we will approximate per-arm reward using
    negative squared error between sim and real volume.
    """

    def __init__(self, arms, init_mean=20.0, init_var=100.0, obs_noise_var=100.0):
        """
        arms: iterable of arm IDs (e.g., tuples (half_bin, sensor, dir))
        init_mean: prior mean for flow
        init_var:  prior variance
        obs_noise_var: assumed observation noise variance
        """
        self.arms = list(arms)
        self.obs_noise_var = obs_noise_var
        self.means = {a: init_mean for a in self.arms}
        self.vars  = {a: init_var  for a in self.arms}

    def sample_flows(self):
        """
        Sample a flow (mean vehicles) for each arm.
        Returns {arm: flow_estimate}
        """
        flows = {}
        for a in self.arms:
            mu = self.means[a]
            var = self.vars[a]
            flows[a] = max(np.random.normal(mu, np.sqrt(var)), 0.0)
        return flows

    def update(self, real_counts, sim_counts, reward=None):
        """
        Update posterior given real and simulated counts for each arm.
        We treat the real volume as (noisy) observation of the mean flow.

        real_counts: {arm: real_volume}
        sim_counts:  {arm: sim_volume}
        """
        # Pick one arm to monitor (Optional)
        debug_arm = next(iter(self.arms)) if self.arms else None
        before = (self.means.get(debug_arm), self.vars.get(debug_arm)) if debug_arm else None

        # Here we only use real_counts to update the prior,
        # but you can also use the error sim - real in a more complex way.
        for a, y in real_counts.items():
            mu_prior = self.means[a]
            var_prior = self.vars[a]
            var_noise = self.obs_noise_var

            # Bayesian update for normal-normal model
            var_post = 1.0 / (1.0 / var_prior + 1.0 / var_noise)
            mu_post = var_post * (mu_prior / var_prior + y / var_noise)

            self.means[a] = mu_post
            self.vars[a]  = var_post

        if debug_arm is not None:
            after = (self.means[debug_arm], self.vars[debug_arm])
            print(f"[BANDIT] Arm {debug_arm} mean/var before {before} -> after {after}")

def build_trips_for_day(flow_estimates, sensor_edge_map, time_bins_per_arm):
    """
    flow_estimates: {arm: flow_value} (arm = (half_bin, sensor, dir))
    sensor_edge_map: {(sensor, dir): (start_edge, end_edge)}
    time_bins_per_arm: {arm: (start_sec, end_sec)} for that half_bin

    Returns list of trips:
        [(veh_id, depart_time, start_edge, end_edge, arm), ...]
    """
    trips = []
    v_idx = 0
    for arm, flow_val in flow_estimates.items():
        half_bin, sensor_id, direction = arm

        # integer number of vehicles
        n = int(round(max(flow_val, 0.0)))

        if n == 0:
            continue

        key = (sensor_id, direction)
        if key not in sensor_edge_map:
            continue  # no mapping, skip

        start_edge, end_edge = sensor_edge_map[key]

        # time interval (in seconds) for this half-hour bin within the day
        t0, t1 = time_bins_per_arm[arm]

        for _ in range(n):
            depart = random.uniform(t0, t1)
            veh_id = f"veh_{v_idx}"
            trips.append((veh_id, depart, start_edge, end_edge, arm))
            v_idx += 1

    print(f"[DEBUG] build_trips_for_day: {len(trips)} trips total.")
    return trips

def build_time_bins_for_half_hour():
    """
    half_bin 0..47 -> (start_sec, end_sec) within a single day.
    """
    mapping = {}
    for hb in range(48):
        start_min = hb * 30
        end_min   = start_min + 30
        mapping[hb] = (start_min * 60, end_min * 60)
    return mapping

def build_time_bins_per_arm(arms):
    """
    arms: iterable of (half_bin, sensor, dir)
    returns {arm: (start_sec, end_sec)}
    """
    half_bin_map = build_time_bins_for_half_hour()
    return {a: half_bin_map[a[0]] for a in arms}

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

def run_sumo_episode(trips, sensor_edge_map, sensor_id_to_detectors):
    """
    trips: [(veh_id, depart, start_edge, end_edge, arm), ...]
    sensor_edge_map: {(sensor, dir): (start_edge, end_edge)}
    sensor_id_to_detectors: {sensor_id: [list_of_sumo_detector_ids]} 
                            Example: {121726: ["det_121726_0", "det_121726_1"]}

    Returns sim_counts: {arm: simulated_volume}
    """
    print(f"[SUMO] Starting episode with {len(trips)} trips.")
    prev_time = time.time()
    
    # Start SUMO
    traci.start([SUMO_BINARY, "-c", SUMO_CFG])

    # Initialize counts for all possible arms to 0
    sim_counts = defaultdict(int)

    try:
        # 1) Add all routes and vehicles (Batch add to be faster)
        for veh_id, depart, start_edge, end_edge, arm in trips:
            route_id = f"r_{veh_id}"
            traci.route.add(route_id, [start_edge, end_edge])
            traci.vehicle.add(
                vehID=veh_id,
                routeID=route_id,
                typeID="car",
                depart=str(depart)
            )

        # 2) Simulation Loop with Sensor Reading
        # Precompute once, before the while-loop
        unique_sensors = {arm[1] for _, _, _, _, arm in trips}
        
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # A. Get current simulation time
            current_time = traci.simulation.getTime()
            
            if step % 600 == 0:
                cur_time = time.time()
                elapsed=cur_time-prev_time
                print(f"[SUMO] step={step}, time={elapsed}, "f"remaining={traci.simulation.getMinExpectedNumber()}")
                prev_time = cur_time


            # B. Determine which "Half-Hour Bin" we are in (0 to 47)
            # 1800 seconds = 30 minutes
            current_half_bin = int(current_time // 1800)
            
            # C. Loop through your sensors and get counts
            # We iterate through the unique sensors in your map
            
            for sensor_id in unique_sensors:
                detector_ids = sensor_id_to_detectors.get(sensor_id, [])
                step_count = 0
                for det_id in detector_ids:
                    step_count += traci.inductionloop.getLastStepVehicleNumber(det_id)

                if step_count > 0:
                    # TODO: use proper direction; placeholder "C" for now
                    arm_key = (current_half_bin, sensor_id, "C")
                    sim_counts[arm_key] += step_count

            step += 1

        print(f"[SUMO] Episode finished at step {step}.")
        print(f"[SUMO] Simulated counts (first 5): {list(sim_counts.items())[:5]}")

    finally:
        traci.close()
        print("[SUMO] Connection closed.")

    return dict(sim_counts)

def compute_loss(sim_counts, real_counts, loss_type="l1"):
    """
    sim_counts:  {arm: simulated_volume}
    real_counts: {arm: real_volume}
    loss_type:   "l1" (absolute error) or "l2" (squared error)

    Returns a scalar loss (higher = worse).
    """
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

        # 1) Bandit samples flow estimates per arm
        flow_estimates = bandit.sample_flows()
        print(f"[DAY {day}] sampled flows (first 5): {list(flow_estimates.items())[:5]}")

        # 2) Build trips for this day from flow estimates
        trips = build_trips_for_day(flow_estimates, sensor_edge_map, time_bins_per_arm)
        print(f"[DAY {day}] generated {len(trips)} trips.")

        # 3) Run SUMO and get simulated counts per arm
        sim_counts = run_sumo_episode(trips, sensor_edge_map, sensor_id_to_detectors)
        
        # 4) Compute simple loss for logging
        loss = compute_loss(sim_counts, real_counts, loss_type="l1")
        reward = -loss
        print(f"[DAY {day}] Loss: {loss:.2f} | Reward: {reward:.2f}")

        # 5) Update bandit with real and simulated counts
        bandit.update(real_counts, sim_counts, reward)
        print(f"[DAY {day}] bandit updated.")

    return bandit

# 2 day test
print(f"Start Training")
train_bandit_with_sumo(train_df, 2)