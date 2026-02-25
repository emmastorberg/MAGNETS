# ChatGPT code for making datasets in .ts format from CSV
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------
# Configuration
# -----------------------

CSV_PATH = "veas_extended_pilot_data_2024.csv"
OUTPUT_TS_BASE = "VeasExtendedPilotData2"

TARGET_COL = "nitrate_out"
WINDOW = 48
HORIZON = 12
TRAIN_RATIO = 0.8  # 80% train, 20% test

FLOAT_FMT = "{:.5f}"   # reduces file size

# -----------------------
# Load data
# -----------------------

df = pd.read_csv(CSV_PATH)

assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found."

# Only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
assert TARGET_COL in numeric_cols, "Target must be numeric."

feature_cols = [c for c in numeric_cols if c != TARGET_COL]

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Total rows: {len(df)}")

data = df[feature_cols].to_numpy(dtype=np.float32)
target = df[TARGET_COL].to_numpy(dtype=np.float32)

N = len(df)
num_samples = N - WINDOW - HORIZON + 1
assert num_samples > 0, "Not enough data for chosen window/horizon."

print(f"Generated samples: {num_samples}")

# -----------------------
# Compute split indices
# -----------------------

n_train = int(num_samples * TRAIN_RATIO)
n_test  = num_samples - n_train

print(f"Train samples: {n_train}")
print(f"Test samples:  {n_test}")

# -----------------------
# Prepare TS headers
# -----------------------

header_lines = [
    f"@problemname {OUTPUT_TS_BASE}\n",
    "@timestamps false\n",
    "@univariate false\n",
    f"@dimensions {len(feature_cols)}\n",
    "@targetLabel true\n",
    "@data\n"
]

# -----------------------
# Prepare output directory
# -----------------------

output_dir = Path(OUTPUT_TS_BASE)
output_dir.mkdir(exist_ok=True)

# -----------------------
# Function to write a .ts file
# -----------------------

def write_ts(path, start_idx, end_idx):
    with open(path, "w") as f:
        f.writelines(header_lines)
        for t in range(start_idx, end_idx):
            dims = []
            for d in range(len(feature_cols)):
                seq = data[t : t + WINDOW, d]
                seq_str = ",".join(FLOAT_FMT.format(x) for x in seq)
                dims.append(seq_str)

            y = target[t + WINDOW + HORIZON - 1]
            row = ":".join(dims) + f":{FLOAT_FMT.format(y)}\n"
            f.write(row)

# -----------------------
# Write train and test .ts files
# -----------------------

write_ts(f"datasets/{output_dir}/{OUTPUT_TS_BASE}_TRAIN.ts", 0, n_train)
write_ts(f"datasets/{output_dir}/{OUTPUT_TS_BASE}_TEST.ts", n_train, num_samples)

print(f"\nSaved:")
print(f" - datasets/{output_dir}/{OUTPUT_TS_BASE}_TRAIN.ts")
print(f" - datasets/{output_dir}/{OUTPUT_TS_BASE}_TEST.ts")