import pandas as pd
import glob
import os

class TrafficDataHandler:
    def __init__(self, data_dir):
        """
        Initialize the TrafficDataHandler.
        
        Args:
            data_dir (str): Path to the directory containing AEDL csv files.
        """
        self.data_dir = data_dir
        self.data = None
        self.pivoted_volume = None

    def load_data(self, pattern="*.csv"):
        """
        Load traffic data from CSV files matching the pattern.
        """
        files = glob.glob(os.path.join(self.data_dir, pattern))
        df_list = []
        
        for file in files:
            # 'sensors_location.csv' is metadata, not traffic data
            if "sensors_location.csv" in file:
                continue
                
            print(f"Loading {file}...")
            try:
                # Use latin-1 or similar if utf-8 fails, based on previous experience
                try:
                    df = pd.read_csv(file, parse_dates=['AGG_PERIOD_START'])
                    print(df.head(5))
                except UnicodeDecodeError:
                    df = pd.read_csv(file, parse_dates=['AGG_PERIOD_START'], encoding='latin-1')
                    

                # Select only relevant columns to save memory
                keep_cols = [
                    'EQUIPMENTID', 'AGG_PERIOD_START', 'TOTAL_VOLUME', 
                    'AVG_SPEED_ARITHMETIC', 'OCCUPANCY'
                ]
                # Filter for columns that actually exist
                existing_cols = [c for c in keep_cols if c in df.columns]
                df = df[existing_cols]
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if df_list:
            self.data = pd.concat(df_list, ignore_index=True)
            print(f"Total records loaded: {len(self.data)}")
        else:
            print("No data loaded.")

    def clean_and_impute(self):
        """
        Clean the data:
        - Handle duplicates
        - Sort by time
        - Interpolate missing values (resample to 5min coverage per sensor)
        """
        if self.data is None:
            print("Data not loaded.")
            return

        # Ensure types
        self.data['AGG_PERIOD_START'] = pd.to_datetime(self.data['AGG_PERIOD_START'])
        self.data['TOTAL_VOLUME'] = pd.to_numeric(self.data['TOTAL_VOLUME'], errors='coerce')
        self.data['AVG_SPEED_ARITHMETIC'] = pd.to_numeric(self.data['AVG_SPEED_ARITHMETIC'], errors='coerce')

        # Drop truly empty rows
        self.data.dropna(subset=['EQUIPMENTID', 'AGG_PERIOD_START'], inplace=True)

        # Sort
        self.data.sort_values(by=['EQUIPMENTID', 'AGG_PERIOD_START'], inplace=True)

        # Remove duplicates
        self.data.drop_duplicates(subset=['EQUIPMENTID', 'AGG_PERIOD_START'], inplace=True)

        # Resample logic via pivoting for efficiency
        print("Starting interpolation...")
        
        # Helper to safely pivot even if dups exist (though we dropped them, sometimes float prec can trick us)
        # We take mean if duplicates still exist
        self.pivoted_volume = self.data.pivot_table(index='AGG_PERIOD_START', columns='EQUIPMENTID', values='TOTAL_VOLUME', aggfunc='mean')
        
        # Interpolate small gaps (limit=3 -> 15 mins)
        self.pivoted_volume.interpolate(method='time', limit=3, inplace=True)
        
        # Fill remaining NaNs with 0 (assuming no traffic if no data for long time? or just keep NaN)
        # For MAB, we might need valid numbers. Let's fill 0 for now or forward fill.
        self.pivoted_volume.fillna(0, inplace=True)
        
        print(f"Data cleaned. Shape: {self.pivoted_volume.shape}")

    def get_data_for_interval(self, start_time, end_time):
        """
        Get data slice.
        """
        if self.pivoted_volume is None:
            return None
        return self.pivoted_volume.loc[start_time:end_time]
