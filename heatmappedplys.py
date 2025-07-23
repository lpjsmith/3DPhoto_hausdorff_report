# import os
# import numpy as np
# import pandas as pd
# import trimesh
# import matplotlib.pyplot as plt
# from scipy.spatial import cKDTree

# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
#     """Computes Hausdorff distances using KDTree and returns statistical metrics."""
#     ref_mesh = trimesh.load_mesh(ref_mesh_path, process=False)
#     target_mesh = trimesh.load_mesh(target_mesh_path, process=False)

#     ref_points = ref_mesh.vertices
#     target_points = target_mesh.vertices

#     kdtree = cKDTree(ref_points)
#     hausdorff_distances, _ = kdtree.query(target_points)

#     metrics = {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": np.min(hausdorff_distances),
#         "Maximum Distance (mm)": np.max(hausdorff_distances),
#         "Mean Distance (mm)": np.mean(hausdorff_distances),
#         "Standard Deviation (mm)": np.std(hausdorff_distances),
#         "RMS Distance (mm)": np.sqrt(np.mean(hausdorff_distances ** 2)),
#         "95th Percentile Distance (mm)": np.percentile(hausdorff_distances, 95),
#         f"Points < {max_dist_threshold}mm": np.sum(hausdorff_distances < max_dist_threshold),
#     }

#     return hausdorff_distances, metrics

# def save_colored_mesh_ply(
#     target_mesh_path,
#     hausdorff_distances,
#     out_path,
#     max_dist_threshold=3.0
# ):
#     mesh = trimesh.load_mesh(target_mesh_path, process=False)

#     face_distances = np.mean(hausdorff_distances[mesh.faces], axis=1)

#     num_bins = 6
#     bins = np.linspace(0, max_dist_threshold, num_bins)

#     # Adjusted desaturated palette for perceptual thirds
#     color_map = np.array([
#         [0, 160, 0],     # Deeper green
#         [90, 200, 50],   # Lighter green close to first bin
#         [210, 210, 0],   # Yellow
#         [255, 180, 60],  # Yellow-orange
#         [255, 110, 60],  # Orange-red
#         [230, 50, 50],   # Lighter red
#     ], dtype=np.uint8)

#     bin_indices = np.digitize(face_distances, bins, right=False)
#     bin_indices = np.clip(bin_indices, 0, len(color_map) - 1)
#     face_colors = color_map[bin_indices]

#     new_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
#     new_faces = np.arange(len(new_vertices)).reshape(-1, 3)
#     expanded_colors = np.repeat(face_colors, 3, axis=0)

#     colored_mesh = trimesh.Trimesh(
#         vertices=new_vertices,
#         faces=new_faces,
#         vertex_colors=expanded_colors,
#         process=False
#     )

#     colored_mesh.export(out_path)
#     print(f"✅ Saved heatmap mesh: {os.path.basename(out_path)}")

# def save_metrics_to_csv(metrics_list, output_csv):
#     df = pd.DataFrame(metrics_list)
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Saved metrics to: {output_csv}")

# def process_mesh_folder(mesh_folder, output_folder, output_csv, max_dist_threshold=3.0):
#     print(f"🟢 Processing folder: {mesh_folder}")
#     os.makedirs(output_folder, exist_ok=True)

#     mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.lower().endswith(('.ply', '.stl', '.obj'))])
#     mesh_paths = {f: os.path.join(mesh_folder, f) for f in mesh_files}

#     all_metrics = []

#     for ref_name in mesh_files:
#         for tgt_name in mesh_files:
#             if ref_name == tgt_name:
#                 continue

#             ref_path = mesh_paths[ref_name]
#             tgt_path = mesh_paths[tgt_name]

#             print(f"🔹 {ref_name} vs {tgt_name}")
#             hausdorff_distances, metrics = compute_hausdorff_metrics(ref_path, tgt_path, max_dist_threshold)
#             all_metrics.append(metrics)

#             out_name = f"heatmap_{ref_name[:7]}_vs_{tgt_name[:7]}.ply"
#             out_path = os.path.join(output_folder, out_name)
#             save_colored_mesh_ply(tgt_path, hausdorff_distances, out_path, max_dist_threshold)

#     save_metrics_to_csv(all_metrics, output_csv)

# if __name__ == "__main__":
#     mesh_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Downloads/2025 Results - july/2025 Results/MHT/craniumpy"
#     output_folder = os.path.join(mesh_folder, "heatmaps")
#     output_csv = os.path.join(mesh_folder, "hausdorff_metrics.csv")
#     max_dist_threshold = 2.0

#     process_mesh_folder(mesh_folder, output_folder, output_csv, max_dist_threshold)
import os
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
    """Computes Hausdorff distances using KDTree and returns statistical metrics."""
    ref_mesh = trimesh.load_mesh(ref_mesh_path, process=False)
    target_mesh = trimesh.load_mesh(target_mesh_path, process=False)

    ref_points = ref_mesh.vertices
    target_points = target_mesh.vertices

    kdtree = cKDTree(ref_points)
    hausdorff_distances, _ = kdtree.query(target_points)

    metrics = {
        "Reference Mesh": os.path.basename(ref_mesh_path),
        "Target Mesh": os.path.basename(target_mesh_path),
        "Minimum Distance (mm)": np.min(hausdorff_distances),
        "Maximum Distance (mm)": np.max(hausdorff_distances),
        "Mean Distance (mm)": np.mean(hausdorff_distances),
        "Standard Deviation (mm)": np.std(hausdorff_distances),
        "RMS Distance (mm)": np.sqrt(np.mean(hausdorff_distances ** 2)),
        "95th Percentile Distance (mm)": np.percentile(hausdorff_distances, 95),
        f"Points < {max_dist_threshold}mm": np.sum(hausdorff_distances < max_dist_threshold),
    }

    return hausdorff_distances, metrics

def generate_scaled_bins(max_dist_threshold, num_bins=8):
    """Generates scaled color bins based on the max distance, keeping evenly spaced intervals."""
    return np.linspace(0, max_dist_threshold, num_bins)

def generate_scaled_colors():
    """Returns the 8-bin RGB color gradient from Green to Red."""
    return np.array([
        [65, 255, 0],  # Green
        [65, 255, 0],  # Green
        [160, 255, 0],  # Light Green
        [250, 255, 0],  # Yellow-Green
        [255, 171, 0],  # Yellow
        [255, 71, 0],  # Light Orange
        [255, 0, 0],  # Red-Orange
        [204, 0, 0],  # Full Red
    ])

def save_colored_mesh_ply(
    target_mesh_path,
    hausdorff_distances,
    out_path,
    max_dist_threshold
):
    mesh = trimesh.load_mesh(target_mesh_path, process=False)

    face_distances = np.mean(hausdorff_distances[mesh.faces], axis=1)

    bins = generate_scaled_bins(max_dist_threshold, num_bins=8)
    color_map = generate_scaled_colors().astype(np.uint8)

    bin_indices = np.digitize(face_distances, bins, right=False)
    bin_indices = np.clip(bin_indices, 0, len(color_map) - 1)
    face_colors = color_map[bin_indices]

    new_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    new_faces = np.arange(len(new_vertices)).reshape(-1, 3)
    expanded_colors = np.repeat(face_colors, 3, axis=0)

    colored_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        vertex_colors=expanded_colors,
        process=False
    )

    colored_mesh.export(out_path)
    print(f"✅ Saved heatmap mesh: {os.path.basename(out_path)}")

def save_metrics_to_csv(metrics_list, output_csv):
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved metrics to: {output_csv}")

def process_mesh_folder(mesh_folder, output_folder, output_csv, max_dist_threshold):
    print(f"🟢 Processing folder: {mesh_folder}")
    os.makedirs(output_folder, exist_ok=True)

    mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.lower().endswith(('.ply', '.stl', '.obj'))])
    mesh_paths = {f: os.path.join(mesh_folder, f) for f in mesh_files}

    all_metrics = []

    for ref_name in mesh_files:
        for tgt_name in mesh_files:
            if ref_name == tgt_name:
                continue

            ref_path = mesh_paths[ref_name]
            tgt_path = mesh_paths[tgt_name]

            print(f"🔹 {ref_name} vs {tgt_name}")
            hausdorff_distances, metrics = compute_hausdorff_metrics(ref_path, tgt_path, max_dist_threshold)
            all_metrics.append(metrics)

            out_name = f"heatmap_{ref_name[:7]}_vs_{tgt_name[:7]}.ply"
            out_path = os.path.join(output_folder, out_name)
            save_colored_mesh_ply(tgt_path, hausdorff_distances, out_path, max_dist_threshold)

    save_metrics_to_csv(all_metrics, output_csv)

if __name__ == "__main__":
    mesh_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Downloads/2025 Results - july/2025 Results/MHT/craniumpy"
    output_folder = os.path.join(mesh_folder, "heatmaps")
    output_csv = os.path.join(mesh_folder, "hausdorff_metrics.csv")
    max_dist_threshold = 2.0

    process_mesh_folder(mesh_folder, output_folder, output_csv, max_dist_threshold)
