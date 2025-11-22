import os
import re
import numpy as np
import pandas as pd
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import pyvista as pv
from trimesh.transformations import rotation_matrix
import pymeshlab as ml  # PyMeshLab for Hausdorff distance

# Enable off-screen rendering for PyVista
os.environ["PYVISTA_OFF_SCREEN"] = "true"
pv.start_xvfb()

# ===== FONT & DRAWING UTILITIES =====
def get_scalable_font(font_size):
    font_paths = [
        "/mnt/c/Windows/Fonts/arial.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "./DejaVuSans-Bold.ttf",
    ]
    for path in font_paths:
        if Path(path).exists():
            print(f"🔤 Using font: {path}")
            return ImageFont.truetype(path, size=font_size)
    print("⚠️ No scalable font found. Using fallback bitmap font.")
    return ImageFont.load_default()


# ===== COLOR SCALE (5-segment: light green → yellow → orange → red) =====
def generate_scaled_bins(min_dist_threshold, max_dist_threshold, num_bins=5):
    """
    Generate (num_bins - 1) evenly spaced internal edges between min and max.
    For 5 colours, gives 4 ranges: <min, 3 internal, >max.
    """
    return np.linspace(min_dist_threshold, max_dist_threshold, num_bins - 1)


def generate_scaled_colors():
    """
    5-segment discrete colour scale:
      - light green (below min)
      - yellow-green
      - light orange (mid)
      - deep orange
      - red (above max)
    """
    return np.array([
        [102, 204, 102],  # lighter green — below min
        [255, 255, 102],  # yellow-green
        [255, 178, 102],  # light orange
        [255, 102, 0],    # deep orange
        [255, 0, 0]       # red — above max
    ])


# ===== HAUSDORFF COMPUTATION =====
def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
    ms = ml.MeshSet()
    ms.load_new_mesh(ref_mesh_path)
    ms.load_new_mesh(target_mesh_path)

    ABS_CUTOFF_MM = 154.6744
    maxdist_arg = ml.PureValue(ABS_CUTOFF_MM) if hasattr(ml, "PureValue") else None

    vcount = ms.mesh(1).vertex_number()
    kwargs = dict(
        sampledmesh=1,
        targetmesh=0,
        savesample=True,
        samplevert=True,
        sampleedge=False,
        sampleface=False,
        samplenum=max(vcount, 1)
    )
    if maxdist_arg is not None:
        kwargs["maxdist"] = maxdist_arg

    try:
        ms.apply_filter("get_hausdorff_distance", **kwargs)
    except TypeError:
        if "maxdist" in kwargs:
            del kwargs["maxdist"]
        ms.apply_filter("get_hausdorff_distance", **kwargs)

    new_ids = list(range(ms.number_meshes()))
    sample_dists = np.asarray(ms.mesh(new_ids[-1]).vertex_scalar_array(), dtype=float)

    if sample_dists.size:
        min_mm, max_mm = float(sample_dists.min()), float(sample_dists.max())
        mean_mm, rms_mm = float(sample_dists.mean()), float(np.sqrt((sample_dists**2).mean()))
    else:
        min_mm = max_mm = mean_mm = rms_mm = 0.0

    ms.apply_filter(
        "compute_scalar_by_distance_from_another_mesh_per_vertex",
        measuremesh=1,
        refmesh=0,
        signeddist=False
    )
    per_vertex_dists = np.asarray(ms.mesh(1).vertex_scalar_array(), dtype=float)

    return per_vertex_dists, {
        "Reference Mesh": os.path.basename(ref_mesh_path),
        "Target Mesh": os.path.basename(target_mesh_path),
        "Minimum Distance (mm)": min_mm,
        "Maximum Distance (mm)": max_mm,
        "Mean Distance (mm)": mean_mm,
        "RMS Distance (mm)": rms_mm,
        "Reference Vertex Count": ms.mesh(0).vertex_number()
    }


# ===== HEATMAP EXPORT =====
def save_colored_mesh_ply(target_mesh_path, hausdorff_distances, out_path,
                          max_dist_threshold, min_dist_threshold=1.0):
    mesh = trimesh.load_mesh(target_mesh_path, process=False)
    face_distances = np.mean(hausdorff_distances[mesh.faces], axis=1)

    bins = generate_scaled_bins(min_dist_threshold, max_dist_threshold, 5)
    color_map = generate_scaled_colors().astype(np.uint8)

    bin_indices = np.digitize(face_distances, bins, right=False)
    bin_indices = np.clip(bin_indices, 0, len(color_map) - 1)

    face_colors = color_map[bin_indices]
    new_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    new_faces = np.arange(len(new_vertices)).reshape(-1, 3)
    expanded_colors = np.repeat(face_colors, 3, axis=0)
    colored_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces,
                                   vertex_colors=expanded_colors, process=False)
    colored_mesh.export(out_path)
    print(f"✅ Saved heatmap mesh: {os.path.basename(out_path)}")


# ===== METRICS CSV =====
def save_metrics_to_csv(metrics_list, output_csv):
    df = pd.DataFrame(metrics_list)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved metrics: {output_csv}")


# ===== UTILS =====
def clean_short_name(filename, max_len=15):
    base = os.path.splitext(filename)[0]
    base = re.sub(r"\W+", "", base)
    return base[:max_len]


# ===== MAIN PIPELINE =====
def process_mesh_folder(mesh_folder, output_folder, output_csv,
                        max_dist_threshold, min_dist_threshold=1.0):
    print(f"🟢 Processing folder: {mesh_folder}")
    os.makedirs(output_folder, exist_ok=True)
    mesh_files = sorted([f for f in os.listdir(mesh_folder)
                         if f.lower().endswith((".ply", ".stl", ".obj"))])
    mesh_paths = {f: os.path.join(mesh_folder, f) for f in mesh_files}

    all_metrics = []
    per_vertex_dists = {clean_short_name(f): [] for f in mesh_files}

    for ref_name in mesh_files:
        for tgt_name in mesh_files:
            if ref_name == tgt_name:
                continue
            ref_path, tgt_path = mesh_paths[ref_name], mesh_paths[tgt_name]
            print(f"🔹 {ref_name} vs {tgt_name}")
            hausdorff_distances, metrics = compute_hausdorff_metrics(ref_path, tgt_path, max_dist_threshold)
            all_metrics.append(metrics)

            tgt_base = clean_short_name(tgt_name)
            per_vertex_dists[tgt_base].append(hausdorff_distances)

            ref_base = clean_short_name(ref_name)
            out_name = f"heatmap_{ref_base}_vs_{tgt_base}.ply"
            out_path = os.path.join(output_folder, out_name)
            save_colored_mesh_ply(tgt_path, hausdorff_distances, out_path,
                                  max_dist_threshold, min_dist_threshold)

    print("\n📊 Averaging vertex-wise Hausdorff distances per mesh...")
    for tgt_name, dist_list in per_vertex_dists.items():
        if not dist_list:
            continue
        stack = np.stack(dist_list, axis=0)
        mean_dists = np.mean(stack, axis=0)
        mae_dists = np.mean(np.abs(stack), axis=0)

        tgt_file = next(f for f in mesh_files if clean_short_name(f) == tgt_name)
        tgt_path = mesh_paths[tgt_file]

        save_colored_mesh_ply(tgt_path, mean_dists,
                              os.path.join(output_folder, f"avg_heatmap_{tgt_name}.ply"),
                              max_dist_threshold, min_dist_threshold)
        save_colored_mesh_ply(tgt_path, mae_dists,
                              os.path.join(output_folder, f"mae_heatmap_{tgt_name}.ply"),
                              max_dist_threshold, min_dist_threshold)
        print(f"✅ Saved mean & MAE heatmaps for {tgt_name}")

    save_metrics_to_csv(all_metrics, output_csv)


# ===== SCREENSHOTS =====
def generate_screenshots_from_ply(ply_folder, only_avg=True):
    pitch_deg, yaw_deg, roll_deg = 50, 0, 135
    for filename in os.listdir(ply_folder):
        if not filename.lower().endswith(".ply"):
            continue
        if only_avg and not (filename.startswith("avg_heatmap_") or filename.startswith("mae_heatmap_")):
            continue
        try:
            ply_path = os.path.join(ply_folder, filename)
            mesh = trimesh.load(ply_path, process=False)
            rotation_center = mesh.centroid
            R_final = (
                rotation_matrix(np.radians(roll_deg), [0, 0, 1]) @
                rotation_matrix(np.radians(yaw_deg), [0, 1, 0]) @
                rotation_matrix(np.radians(pitch_deg), [1, 0, 0])
            )
            T_pre = np.eye(4)
            T_post = np.eye(4)
            T_pre[:3, 3] = -rotation_center
            T_post[:3, 3] = rotation_center
            mesh.apply_transform(T_post @ R_final @ T_pre)

            pv_mesh = pv.wrap(mesh)
            if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
                pv_mesh.cell_data["colors"] = mesh.visual.face_colors[:, :3]

            plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
            plotter.set_background("white")
            plotter.add_mesh(pv_mesh, scalars="colors", rgb=True,
                             show_scalar_bar=False, backface_culling=True)
            plotter.view_vector((1, 1, 1))
            screenshot_path = os.path.join(ply_folder, f"{os.path.splitext(filename)[0]}.png")
            plotter.screenshot(screenshot_path)
            plotter.close()
            print(f"📸 Saved screenshot: {screenshot_path}")
        except Exception as e:
            print(f"❌ Failed to render {filename}: {e}")


# ===== COMBINE AVERAGE/MAE TABLE =====
def combine_avg_mae_table(
    screenshot_folder,
    output_path="combined_table.png",
    min_dist_threshold=0.5,
    max_dist_threshold=1.25
):
    avg_imgs, mae_imgs = {}, {}
    for fname in os.listdir(screenshot_folder):
        fpath = os.path.join(screenshot_folder, fname)
        if fname.startswith("avg_heatmap_") and fname.lower().endswith(".png"):
            key = re.sub(r"^avg_heatmap_|\.png$", "", fname)
            avg_imgs[key] = Image.open(fpath)
        elif fname.startswith("mae_heatmap_") and fname.lower().endswith(".png"):
            key = re.sub(r"^mae_heatmap_|\.png$", "", fname)
            mae_imgs[key] = Image.open(fpath)

    common = sorted(set(avg_imgs.keys()) & set(mae_imgs.keys()))
    if not common:
        print("❌ No matching avg/mae image pairs found.")
        return

    w, h = next(iter(avg_imgs.values())).size
    font = get_scalable_font(int(h * 0.12))
    margin_x, margin_y, legend_h = 120, 60, 90
    text_h = int(h * 0.25)

    table_w = len(common) * w + margin_x
    table_h = 2 * h + 4 * margin_y + text_h + legend_h
    canvas = Image.new("RGB", (table_w, table_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Headers and images
    for i, name in enumerate(common):
        short_name = name[:3].upper()
        x = margin_x + i * w
        text_w, text_h_exact = draw.textbbox((0, 0), short_name, font=font)[2:]
        draw.text((x + (w - text_w)//2, margin_y//2), short_name, fill="black", font=font)
        avg_y = margin_y + text_h
        mae_y = avg_y + h + margin_y
        canvas.paste(avg_imgs[name], (x, avg_y))
        canvas.paste(mae_imgs[name], (x, mae_y))

    row_font = get_scalable_font(int(h * 0.15))
    draw.text((20, margin_y + text_h + h//2 - 20), "Mean", fill="black", font=row_font)
    draw.text((20, margin_y + text_h + h + margin_y + h//2 - 20), "MAE", fill="black", font=row_font)

    # Legend with proper 5 colours and 5 label ranges
    colors = generate_scaled_colors()
    bins = generate_scaled_bins(min_dist_threshold, max_dist_threshold, len(colors))
    n = len(colors)
    seg_w = (table_w - 2 * margin_x) // n
    legend_y = table_h - legend_h - margin_y

    bin_labels = []
    for i in range(n):
        if i == 0:
            bin_labels.append(f"<{min_dist_threshold:.2f}")
        elif i == n - 1:
            bin_labels.append(f">{max_dist_threshold:.2f}")
        else:
            low = bins[i - 1]
            high = bins[i]
            bin_labels.append(f"{low:.2f}–{high:.2f}")

    for i, (color, label) in enumerate(zip(colors, bin_labels)):
        x0 = margin_x + i * seg_w
        x1 = x0 + seg_w
        draw.rectangle([x0, legend_y, x1, legend_y + legend_h//2],
                       fill=tuple(color.tolist()), outline="black")
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.text((x0 + (seg_w - tw)//2, legend_y + legend_h//2 + 5), label, fill="black", font=font)

    canvas.save(output_path)
    print(f"✅ Saved combined table with numeric legend: {output_path}")


# ===== MAIN RUN =====
if __name__ == "__main__":
    mesh_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/properly trimmed-2dl/SNH/SNH NEW/for heatmaps"
    output_folder = os.path.join(mesh_folder, "heatmaps-20251104-mean-mae")
    output_csv = os.path.join(mesh_folder, "Hausdorff_metrics_meshlab.csv")
    
    #MHT
    min_dist_threshold = 0.5
    max_dist_threshold = 1.25

    # #SNH
    # min_dist_threshold = 1.0
    # max_dist_threshold = 2.5


    process_mesh_folder(mesh_folder, output_folder, output_csv,
                        max_dist_threshold, min_dist_threshold)
    generate_screenshots_from_ply(output_folder, only_avg=False)
    combine_avg_mae_table(output_folder,
                          output_path=os.path.join(output_folder, "combined_avg_mae_table.png"),
                          min_dist_threshold=min_dist_threshold,
                          max_dist_threshold=max_dist_threshold)
