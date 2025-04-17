import os
import json
import csv

# Keep the folder path exactly as you provided
folder_path = '/mnt/c/Users/User/Desktop/2025 ERN/2025 ERN/SNH/craniumpy'

# Set CSV file to be saved in the same folder
output_csv = os.path.join(folder_path, "craniometrics.csv")

print("Script started...")  # Debugging

# Ensure the folder exists
if not os.path.exists(folder_path):
    print(f"Error: The folder '{folder_path}' does not exist.")
    exit(1)

# List all JSON files in the folder
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

print(f"Found {len(json_files)} JSON files.")  # Debugging
print(json_files)  # Debugging

if not json_files:
    print("Error: No JSON files found in the folder.")
    exit(1)

# Define the CSV header
csv_header = ["Filename", "OFD_depth_mm", "BPD_breadth_mm", "Cephalic_Index", "Circumference_cm", "MeshVolume_cc"]

# Print absolute path of output CSV
print(f"CSV will be saved to: {output_csv}")

try:
    # Open the CSV file for writing
    with open(output_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)  # Write the header

        # Loop through each JSON file
        for json_file in json_files:
            json_path = os.path.join(folder_path, json_file)
            
            print(f"Processing: {json_file}")  # Debugging statement

            # Read the JSON file
            with open(json_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError as e:
                    print(f"Error reading {json_file}: {e}")
                    continue
            
            print(f"Extracted data from {json_file}: {data}")  # Debugging statement

            # Extract the filename from the "Filepath" field
            filename = os.path.basename(data.get("Filepath", "Unknown"))

            # Check if all required keys exist
            required_keys = ["OFD_depth_mm", "BPD_breadth_mm", "Cephalic_Index", "Circumference_cm", "MeshVolume_cc"]
            if not all(k in data for k in required_keys):
                print(f"Skipping {json_file} due to missing keys.")
                continue

            # Write the data row to the CSV
            csv_writer.writerow([
                filename,
                data["OFD_depth_mm"],
                data["BPD_breadth_mm"],
                data["Cephalic_Index"],
                data["Circumference_cm"],
                data["MeshVolume_cc"]
            ])

    print(f"✅ CSV file saved successfully: {output_csv}")

except Exception as e:
    print(f"❌ Error writing CSV file: {e}")
