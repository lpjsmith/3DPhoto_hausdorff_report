import os
import json
import pandas as pd

# Define the 3 folders
root_path = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/properly trimmed-2dl/MHT/MHT NEW"
folder_names = ["MHT4", "MHT5", "MHT6"]

# Dictionary to store data for each sheet
worksheets_data = {}

for folder_name in folder_names:
    folder_path = os.path.join(root_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    rows = []
    for filename in os.listdir(folder_path):
        if filename.endswith("_metrics.json"):
            full_path = os.path.join(folder_path, filename)
            try:
                with open(full_path, 'r') as f:
                    metrics = json.load(f)
                metrics["source_file"] = filename
                rows.append(metrics)
            except Exception as e:
                print(f"❌ Error reading {full_path}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        worksheets_data[folder_name] = df
    else:
        print(f"⚠️ No metric JSONs found in: {folder_path}")

# Write all to one Excel file
output_file = os.path.join(root_path, "combined_metrics.xlsx")
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    for sheet_name, df in worksheets_data.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet names max = 31 chars

print(f"\n✅ Excel file saved to: {output_file}")
