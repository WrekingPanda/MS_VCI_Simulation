import pandas as pd
import numpy as np
import traci

class TrafficRewardEvaluator:
    def __init__(self, csv_path):
        # Load your dataset
        self.df = pd.read_csv(csv_path)
        
        # PRE-PROCESSING:
        # 1. Convert timestamp to datetime objects for easy filtering
        self.df['AGG_PERIOD_START'] = pd.to_datetime(self.df['AGG_PERIOD_START'])
        
        # 2. Ensure we are looking at the right volume column 
        # (Based on your image, likely 'TOTAL_VOLUME' or 'VOLUME_1')
        self.df.rename(columns={'TOTAL_VOLUME': 'volume', 'AVG_SPEED': 'speed'}, inplace=True)

    def get_real_data(self, sensor_id, current_sim_time_str):
        """
        Fetches the row from the CSV matching the Sensor ID and Time.
        current_sim_time_str format example: '2015-07-24 08:10:00'
        """
        target_time = pd.to_datetime(current_sim_time_str)
        
        # Filter the dataframe
        # Note: Your data has 'LANE_BUNDLE'. We usually sum all lanes for that sensor.
        row = self.df[
            (self.df['EQUIPMENT_ID'] == sensor_id) & 
            (self.df['AGG_PERIOD_START'] == target_time)
        ]
        
        if row.empty:
            return None
            
        # Sum volume across all lanes (C and D bundles) for this time period
        total_real_volume = row['volume'].sum()
        avg_real_speed = row['speed'].mean() # Simplified average
        
        return total_real_volume, avg_real_speed

    def calculate_reward(self, real_vol, sumo_vol, weight_flow=1.0):
        """
        Calculates the score for the Multi-Armed Bandit.
        Returns the Negative Absolute Error (closer to 0 is better).
        """
        # Avoid division by zero issues
        if real_vol == 0 and sumo_vol == 0:
            return 0 
            
        # 1. Absolute Difference (Main objective)
        delta_vol = abs(real_vol - sumo_vol)
        
        # 2. Calculate GEH for logging/validation (Not used for training directly)
        # Formula: sqrt( 2 * (Vol_sim - Vol_real)^2 / (Vol_sim + Vol_real) )
        geh = np.sqrt((2 * (delta_vol**2)) / (real_vol + sumo_vol))
        
        # 3. Construct the Reward
        # We start with 0 and subtract the error. 
        # We can normalize it: e.g., if error is > 50 cars, reward is terrible.
        
        reward = -delta_vol
        
        return reward, geh
    
    
# --- PSEUDO-CODE INSIDE MAIN LOOP ---

evaluator = TrafficRewardEvaluator("your_file.csv")

# A dictionary to count cars over 5 minutes: {sensor_id: count}
interval_accumulators = { "121749": 0, "124680": 0 }

sim_step = 0
while sim_step < 3600:
    traci.simulationStep()
    
    # 1. ACCUMULATE SUMO DATA (Every Step)
    # Get count of cars that passed the detector in the LAST step
    cars_passed_A = traci.inductionloop.getLastStepVehicleNumber("sumo_det_121749")
    interval_accumulators["121749"] += cars_passed_A
    
    # 2. EVALUATE (Every 5 minutes / 300 steps)
    if sim_step > 0 and sim_step % 300 == 0:
        
        # Construct the timestamp string dynamically based on start time
        # E.g., if start is 8:00, and step is 300, current is 8:05
        current_time_str = "2015-07-24 08:05:00" 
        
        # Get Real Data
        real_vol, real_speed = evaluator.get_real_data(121749, current_time_str)
        
        # Get Simulation Data
        sumo_vol = interval_accumulators["121749"]
        
        # Calculate Reward
        reward, geh_score = evaluator.calculate_reward(real_vol, sumo_vol)
        
        print(f"Time: {current_time_str} | Real: {real_vol} | Sim: {sumo_vol} | GEH: {geh_score:.2f}")
        
        # --- UPDATE BANDIT HERE ---
        # my_bandit.update(chosen_arm, reward)
        
        # Reset accumulator for next 5 minutes
        interval_accumulators["121749"] = 0
        
    sim_step += 1