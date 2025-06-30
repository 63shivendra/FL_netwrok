import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("filtered_dataset_condition3500updated.csv")

# Shuffle the DataFrame to ensure full randomization
shuffled_df = df.sample(frac=1)

# Split the shuffled DataFrame into four parts
parts = np.array_split(shuffled_df, 4)

# Save each part to a CSV file with the specified names
for i, part in enumerate(parts, start=1):
    part.reset_index(drop=True, inplace=True)
    part.to_csv(f"hospitaldata{i}.csv", index=False)
