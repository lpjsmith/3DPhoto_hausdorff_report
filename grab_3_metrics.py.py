import os
import json
import pandas as pd
import numpy as np
from scipy.stats import variation

# Path to folder with MHT1, MHT2, MHT3
root_path = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/properly trimmed"

# Find folders
folders = [os.path.join(root_path, f) for f in os.listdir(root_path)
           if os.path.isdir(os.path.join(root_path, f))]
folders.sort()

# Collect data: {metric_file_base: {folder_name: metrics_dict}}
all_data = {}

for folder in folders:
    folder_name = os.path.basename(folder)
    for filename in os.listdir(folder):
        if filename.endswith("_metrics.json"):
            mesh_id = filename.replace("_metrics.json", "")
            full_path = os.path.join(folder, filename)
            with open(full_path) as f:
                metrics = json.load(f)
            if mesh_id not in all_data:
                all_data[mesh_id] = {}
            all_data[mesh_id][folder_name] = metrics

# Flatten to DataFrame
records = []
for mesh_id, folder_metrics in all_data.items():
    for folder_name, metrics in folder_metrics.items():
        row = {"mesh_id": mesh_id, "folder": folder_name}
        row.update(metrics)
        records.append(row)

df = pd.DataFrame(records)
df = df.sort_values(by=["mesh_id", "folder"]).reset_index(drop=True)

# Remove non-metric fields
excluded_fields = {"mesh_id", "folder", "Datetime", "Filepath"}
metric_names = [col for col in df.columns if col not in excluded_fields and np.issubdtype(df[col].dtype, np.number)]

# ICC(2,1) calculation
def calculate_icc(data_matrix):
    n, k = data_matrix.shape
    mean_raters = np.mean(data_matrix, axis=0)
    mean_targets = np.mean(data_matrix, axis=1)
    grand_mean = np.mean(data_matrix)

    MS_between_targets = np.sum((mean_targets - grand_mean) ** 2) * k / (n - 1)
    MS_between_raters = np.sum((mean_raters - grand_mean) ** 2) * n / (k - 1)
    residual = data_matrix - mean_targets[:, None] - mean_raters + grand_mean
    MS_error = np.sum(residual ** 2) / ((k - 1) * (n - 1))

    ICC = (MS_between_targets - MS_error) / (
        MS_between_targets + (k - 1) * MS_error + (k / n) * (MS_between_raters - MS_error)
    )
    return ICC

# Compute ICC and CV
summary_data = []
skipped = []

for metric in metric_names:
    pivot = df.pivot(index="mesh_id", columns="folder", values=metric)

    if pivot.shape[1] < len(folders) or pivot.isnull().values.any():
        skipped.append(metric)
        continue

    values = pivot.values
    icc = calculate_icc(values)
    cv = np.mean(variation(values, axis=1))

    summary_data.append({
        "metric": metric,
        "ICC(2,1)": icc,
        "Coefficient of Variation (mean CV)": cv
    })

# Save and print
summary_df = pd.DataFrame(summary_data)
output_path = os.path.join(root_path, "metrics_reliability_summary.csv")
summary_df.to_csv(output_path, index=False)

print(f"\n✅ Parsed {df['mesh_id'].nunique()} metrics files from {len(folders)} folders.")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} metrics due to missing data:")
    for m in skipped:
        print(f"  - {m}")
print(f"\n📄 Summary saved to: {output_path}")
