import pandas as pd
import numpy as np
import os

# === CONFIGURATION ===
input_csv = "simplified_ehr_data.csv"  # path to your full dataset
output_dir = "hospitals"             # directory where splits will be saved
seed = 42                                      # for reproducible shuffling

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load the full dataset
df = pd.read_csv(input_csv)

# 2. Shuffle the dataframe
df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# 3. Compute split sizes
n = len(df_shuffled)
sizes = [n // 4] * 4
# Distribute any remainder (if n % 4 != 0) across the first few splits
for i in range(n % 4):
    sizes[i] += 1

# 4. Split and save
start = 0
for i, size in enumerate(sizes, start=1):
    end = start + size
    subset = df_shuffled.iloc[start:end]
    out_path = os.path.join(output_dir, f"hospital_{i}.csv")
    subset.to_csv(out_path, index=False)
    print(f"Saved hospital_{i}.csv with {len(subset)} records")
    start = end

print("Done splitting into 4 randomized hospital datasets.")
