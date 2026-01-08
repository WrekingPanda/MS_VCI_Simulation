import pickle
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

TEST_CSV = r"C:\Users\Paulo Alexandre\Documents\PauloAlexandre\Ensino_Superior\MIA\1_Ano\1_Semestre\MS\Projeto\MS_VCI_Simulation\Dataset\simple_test.csv"
MODEL_FILE = "vci_bandit_model.pkl"

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

print("[INFO] Loading test dataset...")
df = pd.read_csv(TEST_CSV, parse_dates=["AGG_PERIOD_START"])

def ensure_half_arm(df):
    if isinstance(df["half_arm"].iloc[0], tuple):
        return df
    df = df.copy()
    df["half_arm"] = df["half_arm"].apply(eval)
    return df

df = ensure_half_arm(df)

df["date"] = df["AGG_PERIOD_START"].dt.date

# ----------------------------------------------------------------------
# LOAD TRAINED MODEL
# ----------------------------------------------------------------------

print("[INFO] Loading trained bandit model...")
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bandit_means = model["means"]

print(f"[INFO] Loaded model with {len(bandit_means)} arms.")

# ----------------------------------------------------------------------
# OFFLINE EVALUATION
# ----------------------------------------------------------------------

metrics = []

for _, row in df.iterrows():
    arm = row["half_arm"]
    real = row["TOTAL_VOLUME"]

    pred = bandit_means.get(arm, 0.0)

    metrics.append({
        "date": row["date"],
        "arm": arm,
        "real": real,
        "pred": pred,
        "abs_err": abs(pred - real),
        "sq_err": (pred - real) ** 2,
        "ape": abs(pred - real) / max(real, 1)
    })

dfm = pd.DataFrame(metrics)

# ----------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------

global_mean = dfm.real.mean()
dfm["baseline_abs_err"] = abs(global_mean - dfm.real)


print("\n================ OFFLINE BANDIT TEST RESULTS ================")
print("Baseline MAE:", dfm.baseline_abs_err.mean())
print(f"MAE  : {dfm.abs_err.mean():.2f}")

daily = dfm.groupby("date")[["real", "pred"]].sum()
daily["abs_err"] = abs(daily.pred - daily.real)
print(f"Daily Absolute MAE: {daily.abs_err.mean():.2f}")
daily["rel_err"] = abs(daily["pred"] - daily["real"]) / daily["real"].replace(0, np.nan)
print("Daily Relative Error:", daily["rel_err"].mean())

print(f"RMSE : {np.sqrt(dfm.sq_err.mean()):.2f}")
nrmse = np.sqrt(dfm.sq_err.mean()) / dfm.real.mean()
print(f"NRMSE: {nrmse:.2f}")

# print(f"MAPE : {dfm.ape.mean() * 100:.2f}%")
smape = (
    2 * abs(dfm.pred - dfm.real) /
    (abs(dfm.real) + abs(dfm.pred)).replace(0, np.nan)
).mean()
print(f"SMAPE: {smape * 100:.2f}%")

wape = dfm.abs_err.sum() / dfm.real.sum()
print(f"WAPE : {wape * 100:.2f}%")
print("============================================================")

print(f"[INFO] Evaluated {len(dfm)} arm observations.")
