import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TEST_CSV = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\Dataset\simple_test.csv"
MODEL_FILE = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\sumoless_bandit_model.pkl"
PROGRESS_PCT = 10   # print every 10%

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("[INFO] Loading test data...")
df = pd.read_csv(TEST_CSV, parse_dates=["AGG_PERIOD_START"])

n_rows = len(df)
print(f"[INFO] Test samples: {n_rows}")

def make_arm(row):
    # return (int(row["half_time_bin"]), int(row["EQUIPMENTID"]))
    # return (int(row["one_time_bin"]), int(row["EQUIPMENTID"]))
    return (int(row["min_since_midnight"]), int(row["EQUIPMENTID"]))

df["arm"] = df.apply(make_arm, axis=1)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("[INFO] Loading trained bandit...")
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bandit_means = model["means"]

print(f"[INFO] Loaded model with {len(bandit_means)} arms.")

# --------------------------------------------------
# AGGREGATE REAL COUNTS
# --------------------------------------------------

print("[INFO] Aggregating real counts...")
real_counts = df.groupby("arm")["TOTAL_VOLUME"].mean().to_dict()

print(f"[INFO] Unique test arms: {len(real_counts)}")

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print("[INFO] Evaluating predictions...")

all_arms = set(real_counts) & set(bandit_means)
records = []

n_arms = len(all_arms)
next_print = PROGRESS_PCT

for i, arm in enumerate(all_arms, start=1):
    real = real_counts.get(arm, 0.0)
    pred = bandit_means.get(arm, 0.0)

    records.append({
        "arm": arm,
        "real": real,
        "pred": pred,
        "abs_err": abs(pred - real),
        "sq_err": (pred - real) ** 2,
        "ape": abs(pred - real) / max(real, 1)
    })

    pct = (i / n_arms) * 100
    if pct >= next_print:
        print(f"    [TEST] {next_print:.0f}% completed ({i}/{n_arms})")
        next_print += PROGRESS_PCT

dfm = pd.DataFrame(records)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

print("\n================ TEST RESULTS ================")
print(f"MAE  : {dfm.abs_err.mean():.2f}")
print(f"RMSE : {np.sqrt(dfm.sq_err.mean()):.2f}")
print(f"MAPE : {dfm.ape.mean() * 100:.2f}%")
print("=============================================")