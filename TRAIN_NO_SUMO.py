import pandas as pd
import pickle
from collections import defaultdict

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TRAIN_CSV = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\Dataset\simple_train.csv"
MODEL_OUT = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\sumoless_bandit_model.pkl"
PROGRESS_PCT = 5   # print every 5%

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("[INFO] Loading training data...")
df = pd.read_csv(TRAIN_CSV, parse_dates=["AGG_PERIOD_START"])

n_rows = len(df)
print(f"[INFO] Training samples: {n_rows}")

# --------------------------------------------------
# DEFINE ARMS
# --------------------------------------------------

def make_arm(row):
    # return (int(row["half_time_bin"]), int(row["EQUIPMENTID"]))
    # return (int(row["one_time_bin"]), int(row["EQUIPMENTID"]))
    return (int(row["min_since_midnight"]), int(row["EQUIPMENTID"]))

df["arm"] = df.apply(make_arm, axis=1)

# --------------------------------------------------
# TRAIN BANDIT
# --------------------------------------------------

arm_sum = defaultdict(float)
arm_count = defaultdict(int)

print("[INFO] Training bandit...")

next_print = PROGRESS_PCT
for i, (_, row) in enumerate(df.iterrows(), start=1):
    arm = row["arm"]
    reward = row["TOTAL_VOLUME"]

    arm_sum[arm] += reward
    arm_count[arm] += 1

    pct = (i / n_rows) * 100
    if pct >= next_print:
        print(f"    [TRAIN] {next_print:.0f}% completed ({i}/{n_rows})")
        next_print += PROGRESS_PCT

# --------------------------------------------------
# FINALIZE MODEL
# --------------------------------------------------

bandit_means = {
    arm: arm_sum[arm] / arm_count[arm]
    for arm in arm_sum
}

print(f"[INFO] Training finished.")
print(f"[INFO] Total arms learned: {len(bandit_means)}")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model = {
    "means": bandit_means,
    "counts": dict(arm_count)
}

with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print(f"[INFO] Model saved to {MODEL_OUT}")