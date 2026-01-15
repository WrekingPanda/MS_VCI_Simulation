import pandas as pd

# Merge and filter data
# -  Read all the sensor data
# - Filter for only data from sensors present in our model
# - Propperly read the timestamps
# - Aggregate all the processed data into a new dataframe

# File names and relative paths
files = ['./Dataset/2013AEDL.csv', './Dataset/1S2014AEDL.csv', './Dataset/2S2014AEDL.csv', './Dataset/1P2015AEDL.csv', './Dataset/2P2015AEDL.csv']

# Define the list of ids from the sensors used in our model
target_ids = [121726,121727,121731,121732,121733,121734,121735,121736,121741,121742,121754,121755,121756]

# Filter the DataFrame: we want to keep only rows where 'EQUIPMENTID' matches the sensor IDs we use
col = 'EQUIPMENTID'

# Create dataframe to concatenate all the useful information
filtered_df = pd.DataFrame()

for file in files:
    df = pd.read_csv(file, parse_dates=["AGG_PERIOD_START"])

    # Convert data types before filtering
    df["AGG_PERIOD_START"] = pd.to_datetime(df["AGG_PERIOD_START"])
    df["LANE_BUNDLE_DIRECTION"] = df["LANE_BUNDLE_DIRECTION"].astype("string")
    
    # Filter for only rows with matching sensor ids
    matching_df = df[df[col].isin(target_ids)]
    
    filtered_df = pd.concat([filtered_df, matching_df], ignore_index=True)

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(matching_df)}")

filtered_df.info()

# Transform data for MAB
## Eliminate useless columns
simple_df = filtered_df.drop(columns=["AGGREGATE_BY_LANE_BUNDLEID","AGG_ID","AGG_PERIOD_LEN_MINS","NR_LANES","AVG_LENGTH","OCCUPANCY","VOLUME_CLASSE_A","VOLUME_CLASSE_B","VOLUME_CLASSE_C","VOLUME_CLASSE_D","VOLUME_CLASSE_0","AXLE_CLASS_VOLUMES","AVG_SPEED_ARITHMETIC","AVG_SPEED_HARMONIC","AVG_SPACING","LIGHT_VEHICLE_RATE"])

## Create tags for arms
### (gather all data from the same sensor, the same time interval and the same direction)

# Map data into arm keys
# Arm = (time_bin, sensor_id, direction)

# Minutes since midnight
simple_df["min_since_midnight"] = (simple_df["AGG_PERIOD_START"].dt.hour*60 + simple_df["AGG_PERIOD_START"].dt.minute)

# 5‑minute bins: [0..287]
simple_df["simple_time_bin"] = (simple_df["min_since_midnight"] // 5).astype("int16")
# This maps 00:00–00:04 → bin 0, 00:05–00:09 → bin 1, …, 23:55–23:59 → bin 287, across all days.

# 30‑minute bins: [0..47]
simple_df["half_time_bin"] = (simple_df["min_since_midnight"] // 30).astype("int8")

# 1 hour bins: [0..23]
simple_df["one_time_bin"] = (simple_df["min_since_midnight"] // 60).astype("int8")

# Define arm identifiers (5-min bin, sensor, direction) as the arm key
simple_df["simple_arm"] = list(
    zip(
        simple_df["simple_time_bin"],
        simple_df["EQUIPMENTID"],
        simple_df["LANE_BUNDLE_DIRECTION"]
    )
)

# Define arm identifiers: (30‑min bin, sensor, direction)
simple_df["half_arm"] = list(
    zip(
        simple_df["half_time_bin"],
        simple_df["EQUIPMENTID"],
        simple_df["LANE_BUNDLE_DIRECTION"]
    )
)

# Define arm identifiers: (1hr bin, sensor, direction)
simple_df["one_arm"] = list(
    zip(
        simple_df["one_time_bin"],
        simple_df["EQUIPMENTID"],
        simple_df["LANE_BUNDLE_DIRECTION"]
    )
)

print("Ammount of arms per time interval aggregation")
simple = simple_df["simple_arm"].nunique()
print(f"5 minutes: {simple}")
half = simple_df["half_arm"].nunique()
print(f"30 minutes: {half}")
one = simple_df["one_arm"].nunique()
print(f"1 hour: {one}")

## Final Preprocessed Data
print(simple_df)
print(simple_df.info())
print(simple_df.describe())

# Train/Test Split
# Separate the train and test data
cutoff_date = pd.Timestamp("2015-01-01")

# Train data: 2013-2014
simple_train_df = simple_df[simple_df["AGG_PERIOD_START"] < cutoff_date]
# Test data: 2015
simple_test_df  = simple_df[simple_df["AGG_PERIOD_START"] >= cutoff_date]

# Save the data
simple_train_df.to_csv("./Dataset/simple_train.csv", index=False)
simple_test_df.to_csv("./Dataset/simple_test.csv", index=False)

## Final Data Splits
print(simple_train_df.info())
print(simple_test_df.info())