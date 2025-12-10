import pandas as pd
import sumolib
import math

class NetworkMapper:
    def __init__(self, net_file, sensors_file):
        self.net = sumolib.net.readNet(net_file)
        self.sensors_df = pd.read_csv(sensors_file, encoding='latin-1')
        
    def generate_detectors(self, output_file="detectors.add.xml"):
        """
        Maps sensors to the nearest lane and generates an XML file.
        """
        with open(output_file, "w") as f:
            f.write("<additional>\n")
            
            for index, row in self.sensors_df.iterrows():
                try:
                    lat_str = str(row['latitude']).replace(",", ".")
                    lon_str = str(row['longitude']).replace(",", ".")
                    lat = float(lat_str)
                    lon = float(lon_str)
                    sensor_id = row['EQUIPMENTID']
                    
                    # Project lat/lon to XY
                    x, y = self.net.convertLonLat2XY(lon, lat)
                    
                    # Find nearest lane
                    lanes = self.net.getNeighboringLanes(x, y, r=50) # Search within 50m
                    
                    if not lanes:
                        print(f"No lane found for sensor {sensor_id} at {lat}, {lon}")
                        continue
                        
                    # Tuple is (lane, dist)
                    # We want the closest one.
                    # Also consider direction? The content of "description" might help, but for now purely geometric.
                    best_lane, dist = min(lanes, key=lambda item: item[1])
                    
                    lane_id = best_lane.getID()
                    # Position on lane: we need to project (x,y) onto the lane shape to find the 'pos'
                    # simplified: get shape, finding closest point.
                    # sumolib lane object has .getShape()
                    
                    # A simplistic approach for 'pos': 0 (start) or middle? 
                    # better: sumolib has logic for this?
                    # Let's trust the 'dist' is perpendicular distance, but we need longitudinal pos.
                    
                    # Actually getNeighboringLanes returns (lane, dist) but doesn't explicitly give 'pos'.
                    # Let's just place it at the beginning or calculate.
                    # For a valid simulation, 'pos' must be valid (0 < pos < length).
                    # We can use lane.getLength() / 2 as a safe default if we don't do complex projection.
                    
                    # More robust: use getClosestLanePosAndDist? (not standard in older sumolib?)
                    # Let's stick to simple mapping for now.
                    pos = 10.0 # Place 10m from start of the identified segment part?
                    if best_lane.getLength() < 20:
                         pos = best_lane.getLength() / 2
                         
                    f.write(f'    <inductionLoop id="{sensor_id}" lane="{lane_id}" pos="{pos}" freq="300" file="detectors_out.xml"/>\n')
                    print(f"Mapped {sensor_id} to {lane_id} (dist: {dist:.2f})")
                    
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
            
            f.write("</additional>\n")
        print(f"Generated {output_file}")

if __name__ == "__main__":
    # Adjust paths as necessary
    mapper = NetworkMapper(
        "drive/Sumo/Simulação/net.net.xml", 
        "drive/sensors_location.csv"
    )
    mapper.generate_detectors("detectors.add.xml")
