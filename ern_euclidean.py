# import os
# import json
# import math
# import pandas as pd

# # Function to calculate Euclidean distance between two points
# def euclidean_distance(point1, point2):
#     return math.sqrt(
#         (point1["x"] - point2["x"]) ** 2 +
#         (point1["y"] - point2["y"]) ** 2 +
#         (point1["z"] - point2["z"]) ** 2
#     )

# # Function to process each subfolder and calculate distances
# def process_landmarks(folder_path, emc_landmarks):
#     distances = []
    
#     # Loop through subfolders
#     for subfolder in os.listdir(folder_path):
#         subfolder_path = os.path.join(folder_path, subfolder)
#         if os.path.isdir(subfolder_path):
#             for file in os.listdir(subfolder_path):
#                 if file.endswith(".txt"):
#                     file_path = os.path.join(subfolder_path, file)
                    
#                     # Read the landmark data from the txt file
#                     with open(file_path, 'r') as f:
#                         landmark_data = json.load(f)
                        
#                     # Calculate distances for matching points
#                     for point_name in emc_landmarks:
#                         if point_name in landmark_data:
#                             distance = euclidean_distance(landmark_data[point_name], emc_landmarks[point_name])
#                             distances.append({"Subfolder": subfolder, "Point": point_name, "Distance": distance})
    
#     return distances

# # Sample EMC landmark coordinates
# emc_landmarks = {
#     "Point01": {
#         "x": 4.968123435974121,
#         "y": -0.004266058094799519,
#         "z": -2.4162721633911133
#     },
#     "Point02": {
#         "x": -5.180057048797607,
#         "y": -0.2189028412103653,
#         "z": -2.4233150482177734
#     },
#     "Point03": {
#         "x": 5.130378246307373,
#         "y": 0.4389784038066864,
#         "z": -2.749807119369507
#     },
#     "Point04": {
#         "x": -5.340620040893555,
#         "y": 0.21686741709709167,
#         "z": -2.827244997024536
#     },
#     "Point05": {
#         "x": 6.157875061035156,
#         "y": -0.00698785949498415,
#         "z": -10.210776329040527
#     },
#     "Point06": {
#         "x": -6.216923713684082,
#         "y": -0.014967020601034164,
#         "z": -10.222124099731445
#     },
#     "Point07": {
#         "x": 2.355830192565918,
#         "y": 0.7344235181808472,
#         "z": -0.4524199962615967
#     },
#     "Point08": {
#         "x": -2.409578561782837,
#         "y": 0.6442674994468689,
#         "z": -0.4731089472770691
#     },
#     "Point09": {
#         "x": 0.05833543837070465,
#         "y": 0.006641482003033161,
#         "z": -0.0016614100895822048
#     },
#     "Point10": {
#         "x": 4.244330406188965,
#         "y": -2.5611042976379395,
#         "z": -2.4118692874908447
#     },
#     "Point11": {
#         "x": -4.372272968292236,
#         "y": -2.5837063789367676,
#         "z": -2.493706464767456
#     },
#     "Point12": {
#         "x": 0.08411543071269989,
#         "y": -5.355294227600098,
#         "z": -1.454758882522583
#     },
#     "Point13": {
#         "x": 6.536006450653076,
#         "y": -1.5557570457458496,
#         "z": -6.442438125610352
#     },
#     "Point14": {
#         "x": -6.555245399475098,
#         "y": -1.6758997440338135,
#         "z": -6.792920112609863
#     },
#     "Point15": {
#         "x": 1.8603096008300781,
#         "y": -7.545230865478516,
#         "z": -3.102368116378784
#     },
#     "Point16": {
#         "x": -1.7317993640899658,
#         "y": -7.367022514343262,
#         "z": -3.7399208545684814
#     }
# }


# # Folder path containing subfolders
# folder_path = "/mnt/c/Users/User/Desktop/ERN wrap landmarks (2)/ERN wrap landmarks/"

# # Process landmarks and create a DataFrame
# distances = process_landmarks(folder_path, emc_landmarks)
# df = pd.DataFrame(distances)

# # Save distances to an Excel file
# output_file = "/mnt/c/Users/User/Desktop/ERN wrap landmarks (2)/landmark_distances.xlsx"
# df.to_excel(output_file, index=False)

# output_file





# Re-importing necessary libraries and redefining the code due to the reset
import os
import json
import math
import pandas as pd

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt(
        (point1["x"] - point2["x"]) ** 2 +
        (point1["y"] - point2["y"]) ** 2 +
        (point1["z"] - point2["z"]) ** 2
    )

# Function to process each subfolder and calculate distances with the requested format
def process_landmarks_with_filename(folder_path, emc_landmarks):
    distances = []
    
    # Loop through subfolders
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(subfolder_path, file)
                    
                    # Read the landmark data from the txt file
                    with open(file_path, 'r') as f:
                        landmark_data = json.load(f)
                        
                    # Calculate distances for matching points
                    for point_name in emc_landmarks:
                        if point_name in landmark_data:
                            distance = euclidean_distance(landmark_data[point_name], emc_landmarks[point_name])
                            distances.append({
                                "Subfolder": subfolder,
                                "Text File": file,
                                "Point Number": point_name,
                                "Distance": distance
                            })
    
    return distances

# Sample EMC landmark coordinates
# Sample EMC landmark coordinates
emc_landmarks = {
    "Point01": {
        "x": -0.011622301302850246,
        "y": 0.16967827081680298,
        "z": -0.030807077884674072
    },
    "Point02": {
        "x": -0.22880768775939941,
        "y": -4.782833099365234,
        "z": 0.8857455253601074
    },
    "Point03": {
        "x": -0.15789103507995605,
        "y": -5.248635768890381,
        "z": -1.2465872764587402
    },
    "Point04": {
        "x": -0.17911027371883392,
        "y": -6.684370994567871,
        "z": -1.4256401062011719
    },
    "Point05": {
        "x": -0.31482911109924316,
        "y": -9.433615684509277,
        "z": -4.205756187438965
    },
    "Point06": {
        "x": 2.1689233779907227,
        "y": -0.41896650195121765,
        "z": -1.5530283451080322
    },
    "Point07": {
        "x": -2.114623546600342,
        "y": -0.350833922624588,
        "z": -1.8703683614730835
    },
    "Point08": {
        "x": 4.910850524902344,
        "y": 0.0165537279099226,
        "z": -2.4909136295318604
    },
    "Point09": {
        "x": -4.966560363769531,
        "y": 0.06155327707529068,
        "z": -2.999687671661377
    },
    "Point10": {
        "x": 7.8523640632629395,
        "y": 0.06839759647846222,
        "z": -10.175008773803711
    },
    "Point11": {
        "x": -7.816011905670166,
        "y": 0.018681026995182037,
        "z": -10.467462539672852
    },
    "Point12": {
        "x": 2.9550137519836426,
        "y": -6.898022651672363,
        "z": -4.393832206726074
    },
    "Point13": {
        "x": -3.450984239578247,
        "y": -6.919828414916992,
        "z": -4.336766719818115
    },
    "Point14": {
        "x": 7.597038269042969,
        "y": 1.857980489730835,
        "z": -9.026557922363281
    },
    "Point15": {
        "x": -7.97442626953125,
        "y": 1.7358580827713013,
        "z": -9.564083099365234
    }
}








# Folder path containing subfolders
folder_path = "/mnt/c/Users/User/Desktop/ERN wrap landmarks (2)/ERN wrap landmarks/"

# Process landmarks and create a DataFrame with the updated format
distances = process_landmarks_with_filename(folder_path, emc_landmarks)
df = pd.DataFrame(distances)

# Save distances to an Excel file
output_file = "/mnt/c/Users/User/Desktop/ERN wrap landmarks (2)/landmark_distances.xlsx"
df.to_excel(output_file, index=False)

output_file
