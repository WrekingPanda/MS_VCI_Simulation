import traci
import sumolib
import os
import sys
import random

class VCISimulation:
    def __init__(self, config_file="drive/Sumo/Simulação/planner.sumo.cfg.xml", 
                 detector_file="detectors.add.xml", 
                 gui=False,
                 mock=False):
        """
        Initialize the VCI Simulation.
        
        Args:
            config_file (str): Path to sumocfg file.
            detector_file (str): Path to additional detectors file.
            gui (bool): Whether to run with sumo-gui or sumo.
            mock (bool): If True, runs in mock mode without SUMO binary.
        """
        self.config_file = config_file
        self.detector_file = detector_file
        self.gui = gui
        self.mock = mock
        self.step_count = 0
        
        if not self.mock:
            # Verify files exist
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
                
            # Determine binary
            try:
                if self.gui:
                    self.sumo_binary = sumolib.checkBinary('sumo-gui')
                else:
                    self.sumo_binary = sumolib.checkBinary('sumo')
            except Exception as e:
                print(f"Warning: SUMO binary check failed: {e}. Falling back to 'sumo'.")
                self.sumo_binary = "sumo"
            
    def start(self):
        """
        Start the simulation using TraCI.
        """
        if self.mock:
            print("MOCK: Simulation started.")
            return

        # Build command line arguments
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.config_file,
            "--additional-files", self.detector_file,
            "--start", "true",
            "--quit-on-end", "true"
        ]
        
        traci.start(sumo_cmd)
        print("Simulation started.")

    def step(self):
        """
        Advance the simulation by one step.
        """
        if self.mock:
            self.step_count += 1
            return

        traci.simulationStep()
        self.step_count += 1
        
    def get_detector_values(self):
        """
        Retrieve vehicle counts and speeds from induction loops.
        Returns:
            dict: {sensor_id: {'count': int, 'speed': float}}
        """
        if self.mock:
             # Return random dummy data for testing the pipeline
             # In a real mock we could use a sine wave or data from the CSV to simulate "perfect" replication
             return {
                 "121725": {'count': random.randint(0, 10), 'speed': random.randint(50, 90)},
                 "121726": {'count': random.randint(0, 15), 'speed': random.randint(60, 100)}
             }

        values = {}
        for detector_id in traci.inductionloop.getIDList():
            # Get last step vehicle number
            count = traci.inductionloop.getLastStepVehicleNumber(detector_id)
            speed = traci.inductionloop.getLastStepMeanSpeed(detector_id)
            values[detector_id] = {'count': count, 'speed': speed}
        return values

    def set_route_flow(self, edge_id, flow_rate):
        """
        Change flow rate for an entrance (for Approach A).
        """
        if self.mock:
            print(f"MOCK: Set flow on {edge_id} to {flow_rate}")
            return
            
        # Implementation would involve traci.edge.set... or spawning vehicles explicitly
        pass

    def close(self):
        """
        Close the simulation.
        """
        if self.mock:
            print("MOCK: Simulation closed.")
            return

        traci.close()
        print("Simulation closed.")

if __name__ == "__main__":
    # Test run
    sim = VCISimulation(gui=False, mock=True)
    sim.start()
    
    # Run for 100 steps
    try:
        for _ in range(100):
            sim.step()
        print(f"Ran 100 steps. Detectors: {len(sim.get_detector_values())}")
    finally:
        sim.close()
