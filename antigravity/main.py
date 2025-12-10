from simulation import VCISimulation
from agents import EntranceAgent, VehicleAgent
from data_loader import TrafficDataHandler
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def run_experiment(approach='B', mock=False, gui=False):
    """
    Run experiment for Approach A or B.
    Args:
        approach (str): 'A' for Entrance Agents, 'B' for Vehicle Agents.
        mock (bool): Dry run mode.
        gui (bool): Whether to run SUMO GUI (requires mock=False).
    """
    print(f"Starting Experiment Approach {approach} (Mock={mock}, GUI={gui})...")
    
    # Load Data for Reward Calculation (Real Mode)
    data_handler = None
    if not mock:
        print("Loading Real Traffic Data...")
        data_handler = TrafficDataHandler("drive/AEDL2013_2015")
        # Load a subset for performance or full dataset
        data_handler.load_data("2013AEDL.csv") 
        data_handler.clean_and_impute()
    
    sim = VCISimulation(mock=mock, gui=gui)
    sim.start()
    
    metrics = {
        'steps': [],
        'rewards': []
    }
    
    try:
        if approach == 'A':
            metrics = run_approach_a(sim, mock, metrics, data_handler)
        elif approach == 'B':
            metrics = run_approach_b(sim, mock, metrics)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        sim.close()
        # Plot results
        plot_results(metrics, approach)

def run_approach_a(sim, mock, metrics, data_handler):
    """
    One Agent per Entrance.
    Controls flow rates.
    """
    entrances = ["edge_input_1", "edge_input_2"]
    action_space = [0.5, 0.8, 1.0, 1.2, 1.5]
    agents = {ent: EntranceAgent(ent, action_space) for ent in entrances}
    
    max_steps = 1000
    aggregation_interval = 300
    
    for step in range(max_steps):
        sim.step()
        
        if step > 0 and step % aggregation_interval == 0:
            print(f"Step {step}: Updating Entrance Agents...")
            # Calculate Reward (RMSE)
            reward = 0
            if mock:
                reward = -np.random.random() 
            else:
                # Compare Simulation vs Real
                # Get current time from simulation (assuming start at 0 corresponds to data start)
                # Ideally we map sim time to real datetime
                # Let's assume simulation starts at 2013-05-09 20:40:00 (from first line of CSV seen)
                start_dt = pd.Timestamp("2013-05-09 20:40:00")
                current_dt = start_dt + pd.Timedelta(seconds=step)
                
                # Get Real Counts for this 5-min interval
                real_data = data_handler.get_data_for_interval(current_dt, current_dt + pd.Timedelta(minutes=5))
                
                sim_data = sim.get_detector_values()
                
                # RMSE Calculation
                squared_errors = []
                if real_data is not None and not real_data.empty:
                    # real_data is a DataFrame row with sensors as columns? 
                    # get_data_for_interval returns DataFrame slice
                    # Our TrafficDataHandler.pivoted_volume columns are EQUIPMENTID
                    
                    # We need to match Sensor IDs (EQUIPMENTID)
                    # Loop through sensors in simulation
                    for sensor_id, vals in sim_data.items():
                         # Check if sensor in real data
                         # Sensor IDs in CSV are integers, SUMO IDs are strings "121725"
                         try:
                             sid = int(sensor_id)
                             if sid in real_data.columns:
                                 # Take mean/sum of real data in this interval
                                 real_val = real_data[sid].mean() # Should be single value if 5min step
                                 sim_val = vals['count']
                                 squared_errors.append((sim_val - real_val) ** 2)
                         except ValueError:
                             pass
                
                if squared_errors:
                    rmse = np.sqrt(np.mean(squared_errors))
                    reward = -rmse
                    print(f"Step {step}: RMSE = {rmse:.2f}")
                else:
                    reward = 0 # No matching data found
                    print(f"Step {step}: No matching data for reward.") 
            
            metrics['steps'].append(step)
            metrics['rewards'].append(reward)
            
            for ent_id, agent in agents.items():
                state = (step // aggregation_interval) % 24
                # update previous action with reward (omitted for brevity, need last_action storage)
                # agent.update(state, last_action, reward)
                
                action = agent.choose_action(state)
                sim.set_route_flow(ent_id, action)
    return metrics

def run_approach_b(sim, mock, metrics):
    """
    One Agent per Vehicle (or homogeneous fleet learning).
    Decides Route.
    """
    od_pairs = [("origin_1", "dest_1"), ("origin_2", "dest_2")]
    routes = {
        ("origin_1", "dest_1"): ["route_1_a", "route_1_b"],
        ("origin_2", "dest_2"): ["route_2_a"]
    }
    
    agents = {}
    for od in od_pairs:
        agents[od] = VehicleAgent(str(od), routes[od])
        
    active_vehicles = {} 
    max_steps = 1000
    
    for step in range(max_steps):
        sim.step()
        
        # MOCK: Simulate vehicle arrival/departure
        if mock:
            # Simulate new vehicle entering
            if step % 10 == 0:
                veh_id = f"veh_{step}"
                od = od_pairs[0]
                agent = agents[od]
                state = 0 
                action = agent.choose_action(state)
                active_vehicles[veh_id] = {'start': step, 'od': od, 'action': action, 'state': state}
            
            # Simulate vehicle arriving
            finished_vehs = [v for v in active_vehicles if step - active_vehicles[v]['start'] > 50]
            for v in finished_vehs:
                data = active_vehicles.pop(v)
                # Add randomness to mock travel time to avoid static -51
                noise = np.random.randint(-5, 6)
                travel_time = step - data['start'] + noise
                reward = -travel_time # Minimize time
                
                # Update Policy
                agent = agents[data['od']]
                agent.update(data['state'], data['action'], reward)
                
                metrics['steps'].append(step)
                metrics['rewards'].append(reward)
    else:
        # Real SUMO Execution for Approach B
        # 1. Detect new vehicles -> Assign Route (Action)
        import traci
        departed_ids = traci.simulation.getDepartedIDList()
        for veh_id in departed_ids:
            # Determine OD (simplified: assume one OD pair for now or read from vehicle)
            # For simplicity, we assign a random OD from our list
            od = od_pairs[0] 
            agent = agents[od]
            
            # State: could be time of day or current congestion
            state = 0 
            action = agent.choose_action(state)
            
            # Assign Route in SUMO
            try:
                traci.vehicle.setRouteID(veh_id, action)
                active_vehicles[veh_id] = {'start': step, 'od': od, 'action': action, 'state': state}
            except Exception as e:
                print(f"Error setting route for {veh_id}: {e}")

        # 2. Detect arriving vehicles -> Reward
        arrived_ids = traci.simulation.getArrivedIDList()
        for veh_id in arrived_ids:
            if veh_id in active_vehicles:
                data = active_vehicles.pop(veh_id)
                travel_time = step - data['start']
                reward = -travel_time
                
                # Update Policy
                agent = agents[data['od']]
                agent.update(data['state'], data['action'], reward)
                
                metrics['steps'].append(step)
                metrics['rewards'].append(reward)
                print(f"Vehicle {veh_id} finished. Time {travel_time}. Reward {reward}")

    return metrics

def plot_results(metrics, approach):
    """
    Plot rewards over time.
    """
    if not metrics['steps']:
        print("No metrics to plot.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['steps'], metrics['rewards'], label='Reward')
    plt.xlabel('Simulation Step')
    plt.ylabel('Reward')
    plt.title(f'Training Results - Approach {approach}')
    plt.legend()
    plt.grid(True)
    
    output_file = f"results_approach_{approach}.png"
    plt.savefig(output_file)
    print(f"Results plotted to {output_file}")

if __name__ == "__main__":
    # To run with SUMO GUI: set mock=False, gui=True
    # To run Mock: set mock=True, gui=False
    # To run with SUMO GUI: set mock=False, gui=True
    # To run Mock: set mock=True, gui=False
    # Setting mock=False to enable real simulation as requested
    run_experiment(approach='B', mock=False, gui=True)
