import subprocess
import xml.etree.ElementTree as ET
import sys

# --- CONFIGURATION ---
ROUTER_FILE = "vci.rou.xml"
CONFIG_FILE = "vci.sumocfg"
FLOW_ID = "flow1"  # The ID of the flow inside your .rou.xml you want to change

def update_flow_in_xml(new_flow_value):
    """
    Opens the XML file, finds the specific flow, changes vehsPerHour, and saves it.
    """
    try:
        tree = ET.parse(ROUTER_FILE)
        root = tree.getroot()
        
        found = False
        # Search for the flow tag with the specific ID
        for flow in root.findall('flow'):
            if flow.get('id') == FLOW_ID:
                print(f"  > Updating '{FLOW_ID}' from {flow.get('vehsPerHour')} to {new_flow_value}...")
                flow.set('vehsPerHour', str(int(new_flow_value)))
                found = True
                break
        
        if not found:
            print(f"Error: Could not find flow with id '{FLOW_ID}' in {ROUTER_FILE}")
            return

        tree.write(ROUTER_FILE)
        print("  > XML file saved successfully.")
        
    except Exception as e:
        print(f"Error updating XML: {e}")

def run_simulation(use_gui=True):
    """
    Runs the SUMO simulation using the system command line.
    """
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    
    # Arguments explained:
    # --autostart: Starts the sim immediately without waiting for you to press Play
    # --quit-on-end: Closes the window automatically when the simulation finishes
    cmd = [sumo_binary, "-c", CONFIG_FILE, "--start", "--quit-on-end", "--delay", "100"]    # delay (measured in ms) allows GUI visualization
    
    print(f"  > Starting SUMO ({sumo_binary})...")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    # Check if SUMO crashed
    if result.returncode != 0:
        print("\nFATAL ERROR: SUMO crashed!")
        print("--------------------------------------------------")
        print("Here is the error message from SUMO:")
        print(result.stderr)  # <--- This prints the actual reason!
        print("--------------------------------------------------")
        exit() # Stop the script so you can read the error
    else:
        print("  > Simulation finished successfully.")
        

# --- MAIN EXPERIMENT LOOP ---
if __name__ == "__main__":
    # Let's try 3 different flow levels to see the difference
    test_values = [400, 800, 1600]

    for i, flow in enumerate(test_values):
        print(f"\n--- Running Experiment {i+1}: Flow = {flow} vehs/hour ---")
        
        # 1. Update the file
        update_flow_in_xml(flow)
        
        # 2. Run the simulation
        # Set use_gui=False if you want it to run fast in the background
        run_simulation(use_gui=True)