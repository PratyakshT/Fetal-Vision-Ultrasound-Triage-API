import pandas as pd
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_csv = os.path.join(base_dir, "data", "raw", "training_labels_formatted.csv")
train_csv = os.path.join(base_dir, "data", "raw", "train_split.csv")
test_csv = os.path.join(base_dir, "data", "raw", "test_split.csv")

# Load and shuffle the data
df = pd.read_csv(raw_csv)
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # random_state ensures reproducibility

# Split 80/20
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Save
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Dataset split successfully!")
print(f"Training set: {len(train_df)} images -> {train_csv}")
print(f"Test set: {len(test_df)} images -> {test_csv}")