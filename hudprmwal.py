import os
import re
import numpy as np
import pandas as pd
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from scipy.spatial import cKDTree
import pyvista as pv
from trimesh.transformations import rotation_matrix
import pymeshlab as ml  # PyMeshLab for distance computation

# Enable off-screen rendering for PyVista
os.environ["PYVISTA_OFF_SCREEN"] = "true"
pv.start_xvfb()

# ===== HEATMAP MATRIX IMAGE FUNCTION =====
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

def draw_colorbar_legend_vertical(width, height, bins, colors, font):
    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)

    num_bins = len(colors)
    segment_height = height // num_bins

    for i, color in enumerate(colors):
        y0 = i * segment_height
        y1 = (i + 1) * segment_height
        draw.rectangle([0, y0, width, y1], fill=tuple(color))

        # Define label
        if i == 0:
            label = f"< {bins[0]:.2f}"
        elif i == len(bins):
            label = f"> {bins[-1]:.2f}"
        else:
            label = f"{bins[i - 1]:.2f} – {bins[i]:.2f}"

        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        text_x = (width - text_w) // 2
        text_y = y0 + (segment_height - text_h) // 2

        text_color = "black" if np.mean(color) < 128 else "black"
        draw.text((text_x, text_y), label, fill=text_color, font=font)

    return legend

def draw_colorbar_legend_horizontal(width, height, bins, colors, font):
    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)

    num_bins = len(colors)
    segment_width = width // num_bins

    for i, color in enumerate(colors):
        x0 = i * segment_width
        x1 = (i + 1) * segment_width
        draw.rectangle([x0, 0, x1, height], fill=tuple(color))

        if i == 0:
            label = f"< {bins[0]:.2f}"
        elif i == len(bins):
            label = f"> {bins[-1]:.2f}"
        else:
            label = f"{bins[i - 1]:.2f} – {bins[i]:.2f}"

        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        text_x = x0 + (segment_width - text_w) // 2
        text_y = (height - text_h) // 2

        text_color = "black" if np.mean(color) < 128 else "black"
        draw.text((text_x, text_y), label, fill=text_color, font=font)

    return legend

def get_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def generate_screenshots_from_ply(ply_folder):
    pitch_deg, yaw_deg, roll_deg = 50, 0, 135

    for filename in os.listdir(ply_folder):
        if filename.lower().endswith('.ply'):
            try:
                ply_path = os.path.join(ply_folder, filename)
                mesh = trimesh.load(ply_path, process=False)

                # Apply rotation around centroid
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
                if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                    pv_mesh.cell_data["colors"] = mesh.visual.face_colors[:, :3]

                plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
                plotter.set_background("white")
                plotter.add_mesh(pv_mesh, scalars="colors", rgb=True, show_scalar_bar=False, backface_culling=True)
                plotter.view_vector((1, 1, 1))

                screenshot_path = os.path.join(ply_folder, f"{os.path.splitext(filename)[0]}.png")
                plotter.screenshot(screenshot_path)
                plotter.close()
                print(f"📸 Saved screenshot: {screenshot_path}")

            except Exception as e:
                print(f"❌ Failed to render: {filename} – {e}")

    # Delete PLY files after screenshot is saved
    for filename in os.listdir(ply_folder):
        if filename.lower().endswith('.ply'):
            try:
                os.remove(os.path.join(ply_folder, filename))
                print(f"🗑️ Deleted: {filename}")
            except Exception as e:
                print(f"⚠️ Could not delete {filename}: {e}")

def create_image_matrix_from_heatmaps(screenshot_folder,
                                      output_path,
                                      colorbar_orientation="horizontal",
                                      min_dist_threshold=1.0,
                                      max_dist_threshold=2.0):

    heatmap_pattern = re.compile(r"heatmap_(.+?)_vs_(.+?)\.png")
    heatmaps = {}
    mesh_names = set()

    for filename in os.listdir(screenshot_folder):
        match = heatmap_pattern.match(filename)
        if match:
            ref, tgt = match.groups()
            heatmaps[(ref, tgt)] = os.path.join(screenshot_folder, filename)
            mesh_names.update([ref, tgt])

    mesh_names = sorted(mesh_names)
    n = len(mesh_names)

    if not heatmaps:
        print("❌ No heatmap PNGs found.")
        return

    sample_img_path = next(iter(heatmaps.values()))
    sample_img = Image.open(sample_img_path)
    cell_w, cell_h = sample_img.size

    matrix_w = (n + 1) * cell_w
    matrix_h = (n + 1) * cell_h
    print(f"🧩 Matrix: {n}×{n}, Cell: {cell_w}×{cell_h}, Canvas: {matrix_w}×{matrix_h}")

    canvas = Image.new("RGB", (matrix_w, matrix_h), "white")
    draw = ImageDraw.Draw(canvas)

    font_size = int(cell_h * 0.15)
    font = get_scalable_font(font_size)

    # Column labels (left side)
    for i, name in enumerate(mesh_names):
        text_w, text_h = get_text_size(draw, name, font)
        x = (cell_w - text_w) // 2
        y = (i + 1) * cell_h + (cell_h - text_h) // 2
        draw.text((x, y), name, fill="black", font=font)

    # Row labels (top)
    for i, name in enumerate(mesh_names):
        text_img = Image.new("RGBA", (cell_w, cell_h), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_w, text_h = get_text_size(text_draw, name, font)
        text_draw.text(((cell_w - text_w) // 2, (cell_h - text_h) // 2), name, fill="black", font=font)
        rotated = text_img.rotate(90, expand=True)
        canvas.paste(rotated, ((i + 1) * cell_w, 0), rotated)

    # Paste heatmaps
    for row_idx, ref in enumerate(mesh_names):
        for col_idx, tgt in enumerate(mesh_names):
            if ref == tgt:
                continue
            key = (ref, tgt)
            if key in heatmaps:
                img = Image.open(heatmaps[key])
                x = (col_idx + 1) * cell_w
                y = (row_idx + 1) * cell_h
                canvas.paste(img, (x, y))

    # Select colorbar orientation
    if colorbar_orientation.lower() == "horizontal":
        colorbar_func = draw_colorbar_legend_horizontal
    elif colorbar_orientation.lower() == "vertical":
        colorbar_func = draw_colorbar_legend_vertical
    else:
        raise ValueError("colorbar_orientation must be 'horizontal' or 'vertical'")

    bins = generate_scaled_bins(min_dist_threshold, max_dist_threshold)
    colors = generate_scaled_colors()
    spacing = 40  # space between matrix and legend

    # Draw and paste color bar
    if colorbar_orientation.lower() == "horizontal":
        colorbar_height = int(cell_h * 0.6)
        legend_img = colorbar_func(canvas.width, colorbar_height, bins, colors, font)
        final_height = canvas.height + spacing + legend_img.height
        final_canvas = Image.new("RGB", (canvas.width, final_height), "white")
        final_canvas.paste(canvas, (0, 0))
        final_canvas.paste(legend_img, (0, canvas.height + spacing))
    else:
        colorbar_width = int(cell_w * 0.6)
        legend_img = colorbar_func(colorbar_width, canvas.height, bins, colors, font)
        final_width = canvas.width + spacing + legend_img.width
        final_canvas = Image.new("RGB", (final_width, canvas.height), "white")
        final_canvas.paste(canvas, (0, 0))
        final_canvas.paste(legend_img, (canvas.width + spacing, 0))

    final_canvas.save(output_path)
    print(f"✅ Saved image matrix with {colorbar_orientation} color bar: {output_path}")

# ===== MAIN MESH ANALYSIS FUNCTIONS =====

# 
# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
#     import pymeshlab as ml
#     ms = ml.MeshSet()
#     ms.load_new_mesh(ref_mesh_path)     # id 0 = reference
#     ms.load_new_mesh(target_mesh_path)  # id 1 = sampled/target

#     # Vertex-only Hausdorff (GUI-equivalent). Avoids random face sampling.
#     try:
#         out = ms.apply_filter(
#             'get_hausdorff_distance',
#             sampledmesh=1,
#             targetmesh=0,
#             savesample=False,
#             samplevert=True,
#             sampleedge=False,
#             sampleface=False,
#             samplenum=1  # kept >=1 for compatibility; ignored since sampleface=False
#         )
#     except TypeError:
#         # Older builds: same call but without optional args that might not exist
#         out = ms.apply_filter(
#             'get_hausdorff_distance',
#             sampledmesh=1,
#             targetmesh=0,
#             samplevert=True,
#             sampleedge=False,
#             sampleface=False,
#             samplenum=1
#         )

#     # Keep generating heatmaps as before (distances in vertex quality of sampled mesh)
#     dists = ms.mesh(1).vertex_scalar_array()
#     if dists is None or len(dists) != ms.mesh(1).vertex_number():
#         # Safety fallback so the rest of your pipeline never breaks
#         ms.apply_filter(
#             'compute_scalar_by_distance_from_another_mesh_per_vertex',
#             measuremesh=1, refmesh=0, signeddist=False
#         )
#         dists = ms.mesh(1).vertex_scalar_array()
#     hausdorff_distances = np.asarray(dists, dtype=float).copy()

#     # Parse GUI-like stats from filter output (robust to key name variants)
#     out = out or {}
#     lower = {k.lower(): v for k, v in out.items()}

#     def pick(*names, fallback=0.0):
#         for n in names:
#             if n in lower and lower[n] is not None:
#                 return float(lower[n])
#         return float(fallback)

#     min_mm  = pick('min', 'mindistance', 'minval', fallback=np.min(hausdorff_distances) if len(hausdorff_distances) else 0.0)
#     max_mm  = pick('max', 'maxdistance', 'maxval', fallback=np.max(hausdorff_distances) if len(hausdorff_distances) else 0.0)
#     mean_mm = pick('mean', 'meandistance', 'meanval', fallback=np.mean(hausdorff_distances) if len(hausdorff_distances) else 0.0)
#     rms_mm  = pick('rms', 'rmsdistance', 'rootmeansquare',
#                    fallback=np.sqrt(np.mean(hausdorff_distances**2)) if len(hausdorff_distances) else 0.0)

#     # Write ONLY the GUI-style Hausdorff stats to CSV (your save_metrics_to_csv will pick these up)
#     return hausdorff_distances, {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": min_mm,
#         "Maximum Distance (mm)": max_mm,
#         "Mean Distance (mm)": mean_mm,
#         "RMS Distance (mm)": rms_mm,
#         "Reference Vertex Count": ms.mesh(0).vertex_number()
#     }

# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
#     import pymeshlab as ml
#     ms = ml.MeshSet()
#     ms.load_new_mesh(ref_mesh_path)     # id 0 = reference
#     ms.load_new_mesh(target_mesh_path)  # id 1 = sampled/target

#     # Vertex-only Hausdorff (GUI-equivalent). No random face sampling.
#     try:
#         ms.apply_filter(
#             'get_hausdorff_distance',
#             sampledmesh=1,
#             targetmesh=0,
#             savesample=False,
#             samplevert=True,
#             sampleedge=False,
#             sampleface=False,
#             samplenum=1  # kept >=1 for compatibility; ignored since sampleface=False
#         )
#     except TypeError:
#         # Older signature without some args
#         ms.apply_filter(
#             'get_hausdorff_distance',
#             sampledmesh=1,
#             targetmesh=0,
#             samplevert=True,
#             sampleedge=False,
#             sampleface=False,
#             samplenum=1
#         )

#     # Pull per-vertex distances from the sampled mesh (id 1).
#     dists = ms.mesh(1).vertex_scalar_array()
#     if dists is None or len(dists) != ms.mesh(1).vertex_number():
#         # Safety fallback so downstream (heatmaps) never breaks
#         ms.apply_filter(
#             'compute_scalar_by_distance_from_another_mesh_per_vertex',
#             measuremesh=1, refmesh=0, signeddist=False
#         )
#         dists = ms.mesh(1).vertex_scalar_array()

#     hausdorff_distances = np.asarray(dists, dtype=float).copy()

#     # Compute GUI-style stats from the vertex-only sampling result
#     if len(hausdorff_distances):
#         min_mm  = float(np.min(hausdorff_distances))
#         max_mm  = float(np.max(hausdorff_distances))
#         mean_mm = float(np.mean(hausdorff_distances))
#         rms_mm  = float(np.sqrt(np.mean(hausdorff_distances ** 2)))
#     else:
#         min_mm = max_mm = mean_mm = rms_mm = 0.0

#     return hausdorff_distances, {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": min_mm,
#         "Maximum Distance (mm)": max_mm,
#         "Mean Distance (mm)": mean_mm,
#         "RMS Distance (mm)": rms_mm,
#         "Reference Vertex Count": ms.mesh(0).vertex_number()
#     }

# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
#     import pymeshlab as ml
#     ms = ml.MeshSet()
#     ms.load_new_mesh(ref_mesh_path)     # id 0 = reference
#     ms.load_new_mesh(target_mesh_path)  # id 1 = sampled/target

#     # GUI-equivalent Hausdorff: vertex-only sampling (no random face samples).
#     try:
#         out = ms.apply_filter(
#             'get_hausdorff_distance',
#             sampledmesh=1,
#             targetmesh=0,
#             savesample=False,
#             samplevert=True,
#             sampleedge=False,
#             sampleface=False,
#             samplenum=1     # kept >=1 for compatibility; ignored since sampleface=False
#         )
#     except TypeError:
#         out = ms.apply_filter(
#             'get_hausdorff_distance',
#             sampledmesh=1,
#             targetmesh=0,
#             samplevert=True,
#             sampleedge=False,
#             sampleface=False,
#             samplenum=1
#         )

#     # ---- Pull GUI-like stats from the filter output (robust to key variants)
#     out = out or {}
#     lower = {str(k).lower(): v for k, v in out.items()}

#     def pick(keys, default=None):
#         for k in keys:
#             if k in lower and lower[k] is not None:
#                 try:
#                     return float(lower[k])
#                 except Exception:
#                     pass
#         return default

#     # Primary: use the dict directly (exactly what the GUI reports)
#     min_mm  = pick(['min', 'mindistance', 'minval'])
#     max_mm  = pick(['max', 'maxdistance', 'maxval'])
#     mean_mm = pick(['mean', 'meandistance', 'meanval'])
#     rms_mm  = pick(['rms', 'rmsdistance', 'rootmeansquare'])

#     # ---- Ensure you have a dense per-vertex field for heatmaps
#     # (independent of the GUI stats so zeros from non-sampled verts won't affect CSV)
#     ms.apply_filter(
#         'compute_scalar_by_distance_from_another_mesh_per_vertex',
#         measuremesh=1, refmesh=0, signeddist=False
#     )
#     dists = ms.mesh(1).vertex_scalar_array()
#     hausdorff_distances = np.asarray(dists, dtype=float).copy()

#     # Fallback: if the dict didn’t expose values, compute from the full per-vertex field
#     if min_mm is None:
#         min_mm = float(np.min(hausdorff_distances)) if len(hausdorff_distances) else 0.0
#     if max_mm is None:
#         max_mm = float(np.max(hausdorff_distances)) if len(hausdorff_distances) else 0.0
#     if mean_mm is None:
#         mean_mm = float(np.mean(hausdorff_distances)) if len(hausdorff_distances) else 0.0
#     if rms_mm is None:
#         rms_mm = float(np.sqrt(np.mean(hausdorff_distances ** 2))) if len(hausdorff_distances) else 0.0

#     return hausdorff_distances, {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": min_mm,
#         "Maximum Distance (mm)": max_mm,
#         "Mean Distance (mm)": mean_mm,
#         "RMS Distance (mm)": rms_mm,
#         "Reference Vertex Count": ms.mesh(0).vertex_number()
#     }

# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
#     import pymeshlab as ml
#     ms = ml.MeshSet()

#     # Keep your existing direction: sampled = TARGET (id 1), target = REFERENCE (id 0)
#     ms.load_new_mesh(ref_mesh_path)      # id 0 = reference (closest-point search target)
#     ms.load_new_mesh(target_mesh_path)   # id 1 = sampled (we sample vertices on this)

#     # Build absolute max-dist argument (version-robust)
#     abs_arg = None
#     if hasattr(ml, "PureValue"):
#         abs_arg = ml.PureValue(154.6744)
#     elif hasattr(ml, "AbsoluteValue"):
#         abs_arg = ml.AbsoluteValue(154.6744)

#     # Run GUI-equivalent Hausdorff with vertex-only sampling and SAVE samples
#     before = ms.number_meshes()
#     kwargs = dict(
#         sampledmesh=1,
#         targetmesh=0,
#         savesample=True,      # ← creates 2 point clouds with distances in vertex quality
#         samplevert=True,      # ← vertices only
#         sampleedge=False,
#         sampleface=False,
#         samplenum=1           # kept >=1 for compatibility; ignored because sampleface=False
#     )
#     if abs_arg is not None:
#         kwargs['maxdist'] = abs_arg

#     try:
#         ms.apply_filter('get_hausdorff_distance', **kwargs)
#     except TypeError:
#         # Some builds don't expose rarely-used args; retry a minimal signature
#         kwargs.pop('maxdist', None)
#         ms.apply_filter('get_hausdorff_distance', **kwargs)

#     after = ms.number_meshes()

#     # Grab the sample cloud distances (first of the two new meshes appended)
#     new_ids = list(range(before, after))
#     sample_cloud_id = new_ids[0] if new_ids else 1
#     sample_dists = np.asarray(ms.mesh(sample_cloud_id).vertex_scalar_array(), dtype=float)

#     # Optional sanity message if maxdist rejected samples (won't stop the run)
#     vcount = ms.mesh(1).vertex_number()
#     if sample_dists.size != vcount:
#         print(f"⚠️ Sample count != vertex count ({sample_dists.size} vs {vcount}) "
#               f"— likely due to maxdist = 154.6744 abs rejecting far samples.")

#     # GUI-style stats computed over the *sample set*
#     if sample_dists.size:
#         min_mm  = float(sample_dists.min())
#         max_mm  = float(sample_dists.max())
#         mean_mm = float(sample_dists.mean())
#         rms_mm  = float(np.sqrt((sample_dists**2).mean()))
#     else:
#         min_mm = max_mm = mean_mm = rms_mm = 0.0

#     # Dense per-vertex field for coloring heatmaps (independent of sample stats)
#     ms.apply_filter(
#         'compute_scalar_by_distance_from_another_mesh_per_vertex',
#         measuremesh=1, refmesh=0, signeddist=False
#     )
#     per_vertex_dists = np.asarray(ms.mesh(1).vertex_scalar_array(), dtype=float)

#     return per_vertex_dists, {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": min_mm,
#         "Maximum Distance (mm)": max_mm,
#         "Mean Distance (mm)": mean_mm,
#         "RMS Distance (mm)": rms_mm,
#         "Reference Vertex Count": ms.mesh(0).vertex_number()
#     }

# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
#     import pymeshlab as ml
#     import trimesh
#     ms = ml.MeshSet()

#     # Load in the same order you want: sampled = TARGET (id 1), target = REFERENCE (id 0)
#     ms.load_new_mesh(ref_mesh_path)      # id 0 = reference (closest-point search target)
#     ms.load_new_mesh(target_mesh_path)   # id 1 = sampled (we sample its vertices)

#     # --- Convert GUI absolute max distance (mm) to Percentage of union bbox diag ---
#     ABS_CUTOFF_MM = 154.6744
#     r = trimesh.load(ref_mesh_path, process=False)
#     t = trimesh.load(target_mesh_path, process=False)
#     mn = np.minimum(r.bounds[0], t.bounds[0])
#     mx = np.maximum(r.bounds[1], t.bounds[1])
#     union_diag = float(np.linalg.norm(mx - mn)) if r.bounds is not None and t.bounds is not None else 0.0
#     if union_diag <= 0:
#         pct = 100.0
#     else:
#         pct = 100.0 * (ABS_CUTOFF_MM / union_diag)

#     # Build Percentage arg for this PyMeshLab version
#     if hasattr(ml, "PercentageValue"):
#         pct_arg = ml.PercentageValue(pct)
#     elif hasattr(ml, "Percentage"):
#         pct_arg = ml.Percentage(pct)
#     else:
#         pct_arg = None  # (very unlikely on 2023.12.post3)

#     # --- Run GUI-equivalent Hausdorff: vertex-only, save samples, with maxdist (as percentage) ---
#     before = ms.number_meshes()
#     kwargs = dict(
#         sampledmesh=1,
#         targetmesh=0,
#         savesample=True,     # creates two point clouds (samples & nearest points), distances in vertex quality
#         samplevert=True,     # vertices only
#         sampleedge=False,
#         sampleface=False,
#         samplenum=1          # required by some builds even if faces aren't sampled
#     )
#     if pct_arg is not None:
#         kwargs['maxdist'] = pct_arg

#     try:
#         ms.apply_filter('get_hausdorff_distance', **kwargs)
#     except TypeError:
#         # If build dislikes an arg, retry minimal signature
#         kwargs.pop('maxdist', None)
#         ms.apply_filter('get_hausdorff_distance', **kwargs)

#     after = ms.number_meshes()

#     # --- Distances from the SAMPLE cloud (first of the two appended meshes) ---
#     new_ids = list(range(before, after))
#     sample_cloud_id = new_ids[0] if new_ids else 1
#     sample_dists = np.asarray(ms.mesh(sample_cloud_id).vertex_scalar_array(), dtype=float)

#     # GUI-style stats over the *sample set*
#     if sample_dists.size:
#         min_mm  = float(sample_dists.min())
#         max_mm  = float(sample_dists.max())
#         mean_mm = float(sample_dists.mean())
#         rms_mm  = float(np.sqrt((sample_dists**2).mean()))
#     else:
#         min_mm = max_mm = mean_mm = rms_mm = 0.0

#     # --- Dense per-vertex field for your heatmaps (independent of sample stats) ---
#     ms.apply_filter(
#         'compute_scalar_by_distance_from_another_mesh_per_vertex',
#         measuremesh=1, refmesh=0, signeddist=False
#     )
#     per_vertex_dists = np.asarray(ms.mesh(1).vertex_scalar_array(), dtype=float)

#     # Optional sanity info
#     vcount = ms.mesh(1).vertex_number()
#     if sample_dists.size != vcount:
#         print(f"ℹ️ GUI maxdist kept {sample_dists.size}/{vcount} vertex-samples "
#               f"({sample_dists.size / max(vcount,1):.1%}) — this is expected if the cutoff removes far points.")
#     print(f"ℹ️ Used maxdist ≈ {pct:.3f}% of union bbox diag ({union_diag:.3f}).")

#     return per_vertex_dists, {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": min_mm,
#         "Maximum Distance (mm)": max_mm,
#         "Mean Distance (mm)": mean_mm,
#         "RMS Distance (mm)": rms_mm,
#         "Reference Vertex Count": ms.mesh(0).vertex_number()
#     }

def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path, max_dist_threshold):
    import pymeshlab as ml
    import numpy as np
    import os

    ms = ml.MeshSet()
    # Keep your direction: sample on TARGET (id 1), search on REFERENCE (id 0)
    ms.load_new_mesh(ref_mesh_path)      # id 0 = reference (closest-point search target)
    ms.load_new_mesh(target_mesh_path)   # id 1 = sampled (we sample its vertices)

    # --- Build maxdist as ABSOLUTE (GUI-style 154.6744) if your build supports it; else convert to % ---
    ABS_CUTOFF_MM = 154.6744
    maxdist_arg = None
    if hasattr(ml, "PureValue"):
        maxdist_arg = ml.PureValue(ABS_CUTOFF_MM)
    elif hasattr(ml, "AbsoluteValue"):
        maxdist_arg = ml.AbsoluteValue(ABS_CUTOFF_MM)

    # If absolute types not available, we’ll omit maxdist here (no drop) and
    # you can keep the earlier percentage-conversion version if needed.

    vcount = ms.mesh(1).vertex_number()  # <-- drive sample count by vertex count

    before = ms.number_meshes()
    kwargs = dict(
        sampledmesh=1,
        targetmesh=0,
        savesample=True,      # create two point clouds (samples & nearest points)
        samplevert=True,      # vertices ON
        sampleedge=False,
        sampleface=False,     # no MC face sampling
        samplenum=max(vcount, 1)  # <-- use vertex count to request that many samples
    )
    if maxdist_arg is not None:
        kwargs['maxdist'] = maxdist_arg

    # First try with full signature (some builds are picky about unknown args)
    try:
        ms.apply_filter('get_hausdorff_distance', **kwargs)
    except TypeError:
        # Retry without maxdist if absolute type isn't supported
        if 'maxdist' in kwargs:
            del kwargs['maxdist']
        ms.apply_filter('get_hausdorff_distance', **kwargs)

    after = ms.number_meshes()

    # Distances from the SAMPLE cloud (first of the two appended meshes)
    new_ids = list(range(before, after))
    sample_cloud_id = new_ids[0] if new_ids else 1
    sample_dists = np.asarray(ms.mesh(sample_cloud_id).vertex_scalar_array(), dtype=float)

    # NOTE: with an active maxdist, samples farther than the cutoff are REJECTED by the filter,
    # so len(sample_dists) can be < vcount. That’s expected and mirrors the GUI.
    if sample_dists.size != vcount:
        print(f"ℹ️ Kept {sample_dists.size}/{vcount} vertex-samples after maxdist filtering.")

    # GUI-style stats over the *accepted sample set*
    if sample_dists.size:
        min_mm  = float(sample_dists.min())
        max_mm  = float(sample_dists.max())
        mean_mm = float(sample_dists.mean())
        rms_mm  = float(np.sqrt((sample_dists**2).mean()))
    else:
        min_mm = max_mm = mean_mm = rms_mm = 0.0

    # Dense per-vertex field for coloring heatmaps (independent of sample stats)
    ms.apply_filter(
        'compute_scalar_by_distance_from_another_mesh_per_vertex',
        measuremesh=1, refmesh=0, signeddist=False
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


def generate_scaled_bins(min_dist_threshold, max_dist_threshold, num_bins=3):
    return np.linspace(min_dist_threshold, max_dist_threshold, num_bins-1)

def generate_scaled_colors():
    return np.array([
        [65, 255, 0],
        [255, 255, 0],
        [255, 0, 0]
    ])

def save_colored_mesh_ply(target_mesh_path, hausdorff_distances, out_path, max_dist_threshold):
    mesh = trimesh.load_mesh(target_mesh_path, process=False)
    face_distances = np.mean(hausdorff_distances[mesh.faces], axis=1)
    bins = generate_scaled_bins(min_dist_threshold, max_dist_threshold)
    color_map = generate_scaled_colors().astype(np.uint8)
    bin_indices = np.clip(np.digitize(face_distances, bins, right=False), 0, len(color_map) - 1)
    face_colors = color_map[bin_indices]
    new_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    new_faces = np.arange(len(new_vertices)).reshape(-1, 3)
    expanded_colors = np.repeat(face_colors, 3, axis=0)
    colored_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, vertex_colors=expanded_colors, process=False)
    colored_mesh.export(out_path)
    print(f"✅ Saved heatmap mesh: {os.path.basename(out_path)}")

def save_metrics_to_csv(metrics_list, output_csv):
    df = pd.DataFrame(metrics_list)
    df.reset_index(drop=True, inplace=True)

    weight_col = "Reference Vertex Count"
    weight_col_idx = df.columns.get_loc(weight_col)

    # Column names
    mean_col = "Mean Distance (mm)"
    std_col = "Standard Deviation (mm)"
    rms_col = "RMS Distance (mm)"
    perc95_col = "95th Percentile Distance (mm)"

    numeric_cols = df.select_dtypes(include=np.number).columns

    # Excel row numbers (data starts on row 2 in Excel)
    start_row = 2
    end_row = start_row + len(df) - 1

    def col_letter(idx):
        result = ""
        while idx >= 0:
            result = chr(idx % 26 + ord('A')) + result
            idx = idx // 26 - 1
        return result

    formula_row = {
        "Reference Mesh": "Weighted Summary",
        "Target Mesh": "",
        weight_col: f"=SUM({col_letter(weight_col_idx)}{start_row}:{col_letter(weight_col_idx)}{end_row})"
    }

    for col in numeric_cols:
        col_idx = df.columns.get_loc(col)
        col_L = col_letter(col_idx)
        weight_L = col_letter(weight_col_idx)

    # Weighted mean, RMS; population-like weighted std; simple avg for P95 (proxy)
        if col == mean_col:
            formula = (
                f"=SUMPRODUCT({col_L}{start_row}:{col_L}{end_row},"
                f"{weight_L}{start_row}:{weight_L}{end_row}) / "
                f"SUM({weight_L}{start_row}:{weight_L}{end_row})"
            )
        elif col == std_col:
            formula = (
                f"=SQRT(SUMPRODUCT(({col_L}{start_row}:{col_L}{end_row} - "
                f"AVERAGE({col_L}{start_row}:{col_L}{end_row}))^2, "
                f"{weight_L}{start_row}:{weight_L}{end_row}) / "
                f"SUM({weight_L}{start_row}:{weight_L}{end_row}))"
            )
        elif col == rms_col:
            formula = (
                f"=SQRT(SUMPRODUCT(({col_L}{start_row}:{col_L}{end_row})^2, "
                f"{weight_L}{start_row}:{weight_L}{end_row}) / "
                f"SUM({weight_L}{start_row}:{weight_L}{end_row}))"
            )
        elif col == perc95_col:
            formula = f"=AVERAGE({col_L}{start_row}:{col_L}{end_row})"
        else:
            continue  # Skip other numeric columns

        formula_row[col] = formula

    # Fill blanks for any unused columns
    for col in df.columns:
        if col not in formula_row:
            formula_row[col] = ""

    df = pd.concat([df, pd.DataFrame([formula_row])], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved metrics with Excel formulas to: {output_csv}")

def clean_short_name(filename, max_len=15):
    base = os.path.splitext(filename)[0]
    base = re.sub(r'\W+', '', base)  # Remove non-alphanumerics
    return base[:max_len]

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
            ref_path, tgt_path = mesh_paths[ref_name], mesh_paths[tgt_name]
            print(f"🔹 {ref_name} vs {tgt_name}")
            hausdorff_distances, metrics = compute_hausdorff_metrics(ref_path, tgt_path, max_dist_threshold)
            all_metrics.append(metrics)
            ref_base = clean_short_name(ref_name)
            tgt_base = clean_short_name(tgt_name)
            out_name = f"heatmap_{ref_base}_vs_{tgt_base}.ply"
            out_path = os.path.join(output_folder, out_name)
            save_colored_mesh_ply(tgt_path, hausdorff_distances, out_path, max_dist_threshold)
    save_metrics_to_csv(all_metrics, output_csv)

def generate_screenshots_from_ply(ply_folder):
    pitch_deg, yaw_deg, roll_deg = 50, 0, 135

    for filename in os.listdir(ply_folder):
        if filename.lower().endswith('.ply'):
            try:
                ply_path = os.path.join(ply_folder, filename)
                mesh = trimesh.load(ply_path, process=False)

                # Apply rotation around centroid
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
                if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                    pv_mesh.cell_data["colors"] = mesh.visual.face_colors[:, :3]

                plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
                plotter.set_background("white")
                plotter.add_mesh(pv_mesh, scalars="colors", rgb=True, show_scalar_bar=False, backface_culling=True)
                plotter.view_vector((1, 1, 1))

                screenshot_path = os.path.join(ply_folder, f"{os.path.splitext(filename)[0]}.png")
                plotter.screenshot(screenshot_path)
                plotter.close()
                print(f"📸 Saved screenshot: {screenshot_path}")

            except Exception as e:
                print(f"❌ Failed to render: {filename} – {e}")

    # Delete PLY files after screenshot is saved
    for filename in os.listdir(ply_folder):
        if filename.lower().endswith('.ply'):
            try:
                os.remove(os.path.join(ply_folder, filename))
                print(f"🗑️ Deleted: {filename}")
            except Exception as e:
                print(f"⚠️ Could not delete {filename}: {e}")

# ===== MAIN RUN BLOCK =====
if __name__ == "__main__":
    mesh_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/properly trimmed-2dl/MHT/MHT OLD/MHT1/best ones"
    output_folder = os.path.join(mesh_folder, "heatmaps-20251104")
    output_csv = os.path.join(mesh_folder, "Hausdorff_metrics_meshlab.csv")
    min_dist_threshold = 0.5
    max_dist_threshold = 1.0

    process_mesh_folder(mesh_folder, output_folder, output_csv, max_dist_threshold)
    generate_screenshots_from_ply(output_folder)

    screenshot_folder = output_folder
    image_matrix_path = os.path.join(screenshot_folder, "image_matrix_biglabels.png")
    # create_image_matrix_from_heatmaps(screenshot_folder, image_matrix_path)
    create_image_matrix_from_heatmaps(screenshot_folder,
                                      image_matrix_path,
                                      colorbar_orientation="horizontal",
                                      min_dist_threshold=min_dist_threshold,
                                      max_dist_threshold=max_dist_threshold
                                      )
