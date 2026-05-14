# import os
# import re
# import numpy as np
# import pandas as pd
# import trimesh
# from PIL import Image, ImageDraw, ImageFont
# from pathlib import Path
# import pyvista as pv
# from trimesh.transformations import rotation_matrix
# import pymeshlab as ml  # PyMeshLab for Hausdorff distance

# # Enable off-screen rendering for PyVista
# os.environ["PYVISTA_OFF_SCREEN"] = "true"
# pv.start_xvfb()

# # ===== FONT & DRAWING UTILITIES =====
# def get_scalable_font(font_size: int):
#     font_paths = [
#         "/mnt/c/Windows/Fonts/arial.ttf",
#         "arial.ttf",
#         "C:/Windows/Fonts/arial.ttf",
#         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
#         "./DejaVuSans-Bold.ttf",
#     ]
#     for path in font_paths:
#         if Path(path).exists():
#             print(f"🔤 Using font: {path}")
#             return ImageFont.truetype(path, size=font_size)
#     print("⚠️ No scalable font found. Using fallback bitmap font.")
#     return ImageFont.load_default()


# # ===== FIXED COLOR SCALE (4 bins: green/yellow/orange/red) =====
# def get_fixed_bins_mm():
#     # Internal edges for np.digitize -> 4 bins:
#     # idx 0: <1, idx 1: [1,2), idx 2: [2,3), idx 3: >=3
#     return np.array([1.0, 2.0, 3.0], dtype=float)


# def get_fixed_colors_rgb():
#     # Order corresponds to bins: <1, 1–2, 2–3, >=3
#     return np.array([
#         [0, 200, 0],      # green
#         [255, 255, 0],    # yellow
#         [255, 165, 0],    # orange
#         [255, 0, 0],      # red
#     ], dtype=np.uint8)


# # ===== HAUSDORFF COMPUTATION =====
# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path):
#     ms = ml.MeshSet()
#     ms.load_new_mesh(ref_mesh_path)     # mesh 0
#     ms.load_new_mesh(target_mesh_path)  # mesh 1

#     # Optional absolute cutoff to prevent crazy values blowing things up
#     ABS_CUTOFF_MM = 154.6744
#     maxdist_arg = ml.PureValue(ABS_CUTOFF_MM) if hasattr(ml, "PureValue") else None

#     vcount = ms.mesh(1).vertex_number()
#     kwargs = dict(
#         sampledmesh=1,
#         targetmesh=0,
#         savesample=True,
#         samplevert=True,
#         sampleedge=False,
#         sampleface=False,
#         samplenum=max(vcount, 1)
#     )
#     if maxdist_arg is not None:
#         kwargs["maxdist"] = maxdist_arg

#     try:
#         ms.apply_filter("get_hausdorff_distance", **kwargs)
#     except TypeError:
#         # Some builds don't accept maxdist
#         kwargs.pop("maxdist", None)
#         ms.apply_filter("get_hausdorff_distance", **kwargs)

#     # Hausdorff sample mesh is appended at the end
#     new_ids = list(range(ms.number_meshes()))
#     sample_dists = np.asarray(ms.mesh(new_ids[-1]).vertex_scalar_array(), dtype=float)

#     if sample_dists.size:
#         min_mm = float(sample_dists.min())
#         max_mm = float(sample_dists.max())
#         mean_mm = float(sample_dists.mean())
#         rms_mm = float(np.sqrt((sample_dists ** 2).mean()))
#     else:
#         min_mm = max_mm = mean_mm = rms_mm = 0.0

#     # Per-vertex distance on mesh(1) from ref mesh(0)
#     ms.apply_filter(
#         "compute_scalar_by_distance_from_another_mesh_per_vertex",
#         measuremesh=1,
#         refmesh=0,
#         signeddist=False
#     )
#     per_vertex_dists = np.asarray(ms.mesh(1).vertex_scalar_array(), dtype=float)

#     metrics = {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": min_mm,
#         "Maximum Distance (mm)": max_mm,
#         "Mean Distance (mm)": mean_mm,
#         "RMS Distance (mm)": rms_mm,
#         "Reference Vertex Count": ms.mesh(0).vertex_number()
#     }
#     return per_vertex_dists, metrics


# # ===== HEATMAP EXPORT =====
# def save_colored_mesh_ply(target_mesh_path, hausdorff_distances, out_path):
#     mesh = trimesh.load_mesh(target_mesh_path, process=False)

#     # Face distance = mean of the vertex distances on that face
#     face_distances = np.mean(hausdorff_distances[mesh.faces], axis=1)

#     bins = get_fixed_bins_mm()
#     color_map = get_fixed_colors_rgb()

#     # 0 for <1, 1 for [1,2), 2 for [2,3), 3 for >=3
#     bin_indices = np.digitize(face_distances, bins, right=False)
#     bin_indices = np.clip(bin_indices, 0, len(color_map) - 1)

#     face_colors = color_map[bin_indices]

#     # Explode faces so each triangle can have a flat colour
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


# # ===== METRICS CSV =====
# def save_metrics_to_csv(metrics_list, output_csv):
#     df = pd.DataFrame(metrics_list)
#     df.reset_index(drop=True, inplace=True)
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Saved metrics: {output_csv}")


# # ===== UTILS =====
# def clean_short_name(filename, max_len=15):
#     base = os.path.splitext(filename)[0]
#     base = re.sub(r"\W+", "", base)
#     return base[:max_len]


# # ===== MAIN PIPELINE =====
# def process_mesh_folder(mesh_folder, output_folder, output_csv):
#     print(f"🟢 Processing folder: {mesh_folder}")
#     os.makedirs(output_folder, exist_ok=True)

#     mesh_files = sorted([f for f in os.listdir(mesh_folder)
#                          if f.lower().endswith((".ply", ".stl", ".obj"))])
#     mesh_paths = {f: os.path.join(mesh_folder, f) for f in mesh_files}

#     all_metrics = []
#     per_vertex_dists = {clean_short_name(f): [] for f in mesh_files}

#     for ref_name in mesh_files:
#         for tgt_name in mesh_files:
#             if ref_name == tgt_name:
#                 continue

#             ref_path, tgt_path = mesh_paths[ref_name], mesh_paths[tgt_name]
#             print(f"🔹 {ref_name} vs {tgt_name}")

#             hausdorff_distances, metrics = compute_hausdorff_metrics(ref_path, tgt_path)
#             all_metrics.append(metrics)

#             tgt_base = clean_short_name(tgt_name)
#             per_vertex_dists[tgt_base].append(hausdorff_distances)

#             ref_base = clean_short_name(ref_name)
#             out_name = f"heatmap_{ref_base}_vs_{tgt_base}.ply"
#             out_path = os.path.join(output_folder, out_name)
#             save_colored_mesh_ply(tgt_path, hausdorff_distances, out_path)

#     print("\n📊 Averaging vertex-wise Hausdorff distances per mesh...")
#     for tgt_name, dist_list in per_vertex_dists.items():
#         if not dist_list:
#             continue

#         stack = np.stack(dist_list, axis=0)
#         mean_dists = np.mean(stack, axis=0)
#         mae_dists = np.mean(np.abs(stack), axis=0)

#         tgt_file = next(f for f in mesh_files if clean_short_name(f) == tgt_name)
#         tgt_path = mesh_paths[tgt_file]

#         save_colored_mesh_ply(
#             tgt_path, mean_dists,
#             os.path.join(output_folder, f"avg_heatmap_{tgt_name}.ply")
#         )
#         save_colored_mesh_ply(
#             tgt_path, mae_dists,
#             os.path.join(output_folder, f"mae_heatmap_{tgt_name}.ply")
#         )
#         print(f"✅ Saved mean & MAE heatmaps for {tgt_name}")

#     save_metrics_to_csv(all_metrics, output_csv)


# # ===== SCREENSHOTS =====
# def generate_screenshots_from_ply(ply_folder, only_avg=True):
#     pitch_deg, yaw_deg, roll_deg = 50, 0, 135

#     for filename in os.listdir(ply_folder):
#         if not filename.lower().endswith(".ply"):
#             continue
#         if only_avg and not (filename.startswith("avg_heatmap_") or filename.startswith("mae_heatmap_")):
#             continue

#         try:
#             ply_path = os.path.join(ply_folder, filename)
#             mesh = trimesh.load(ply_path, process=False)

#             rotation_center = mesh.centroid
#             R_final = (
#                 rotation_matrix(np.radians(roll_deg), [0, 0, 1]) @
#                 rotation_matrix(np.radians(yaw_deg), [0, 1, 0]) @
#                 rotation_matrix(np.radians(pitch_deg), [1, 0, 0])
#             )
#             T_pre = np.eye(4)
#             T_post = np.eye(4)
#             T_pre[:3, 3] = -rotation_center
#             T_post[:3, 3] = rotation_center
#             mesh.apply_transform(T_post @ R_final @ T_pre)

#             pv_mesh = pv.wrap(mesh)

#             if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
#                 pv_mesh.cell_data["colors"] = mesh.visual.face_colors[:, :3]
#                 scalars_name = "colors"
#                 rgb = True
#                 use_cell = True
#             elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
#                 pv_mesh.point_data["colors"] = mesh.visual.vertex_colors[:, :3]
#                 scalars_name = "colors"
#                 rgb = True
#                 use_cell = False
#             else:
#                 scalars_name = None
#                 rgb = False
#                 use_cell = False

#             plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
#             plotter.set_background("white")

#             if scalars_name is None:
#                 plotter.add_mesh(pv_mesh, show_scalar_bar=False, backface_culling=True)
#             else:
#                 plotter.add_mesh(
#                     pv_mesh,
#                     scalars=scalars_name,
#                     rgb=rgb,
#                     show_scalar_bar=False,
#                     backface_culling=True,
#                     preference="cell" if use_cell else "point"
#                 )

#             plotter.view_vector((1, 1, 1))
#             screenshot_path = os.path.join(ply_folder, f"{os.path.splitext(filename)[0]}.png")
#             plotter.screenshot(screenshot_path)
#             plotter.close()
#             print(f"📸 Saved screenshot: {screenshot_path}")

#         except Exception as e:
#             print(f"❌ Failed to render {filename}: {e}")


# # ===== COMBINE AVERAGE/MAE TABLE (CUSTOM ORDER + LABELS) =====
# def combine_avg_mae_table(
#     screenshot_folder,
#     output_path="combined_table.png"
# ):
#     avg_imgs, mae_imgs = {}, {}

#     for fname in os.listdir(screenshot_folder):
#         fpath = os.path.join(screenshot_folder, fname)
#         if fname.startswith("avg_heatmap_") and fname.lower().endswith(".png"):
#             key = re.sub(r"^avg_heatmap_|\.png$", "", fname)
#             avg_imgs[key] = Image.open(fpath)
#         elif fname.startswith("mae_heatmap_") and fname.lower().endswith(".png"):
#             key = re.sub(r"^mae_heatmap_|\.png$", "", fname)
#             mae_imgs[key] = Image.open(fpath)

#     common_all = sorted(set(avg_imgs.keys()) & set(mae_imgs.keys()))
#     if not common_all:
#         print("❌ No matching avg/mae image pairs found.")
#         return

#     # Desired left-to-right order (supports duplicates like nck twice)
#     desired_order = ["emc", "gos", "rad", "sah", "ukt", "cha", "nck"]
#     desired_labels = ["A_3dMD", "B_3dMD", "C_3dMD", "D_3dMD", "E_Vectra", "F_Vectra", "G_Artec"]

#     common_ordered = []
#     used = set()
#     for code in desired_order:
#         match = None
#         for k in common_all:
#             if k in used:
#                 continue
#             if k.lower().startswith(code.lower()):
#                 match = k
#                 break
#         if match is None:
#             print(f"⚠️ Missing image pair for code '{code}' (no unused key starting with '{code}').")
#             continue
#         used.add(match)
#         common_ordered.append(match)

#     if not common_ordered:
#         print("❌ None of the requested ordered codes were found as image pairs.")
#         return

#     # Layout
#     w, h = next(iter(avg_imgs.values())).size
#     font = get_scalable_font(int(h * 0.12))
#     margin_x, margin_y, legend_h = 120, 60, 90
#     header_text_h = int(h * 0.25)

#     table_w = len(common_ordered) * w + margin_x
#     table_h = 2 * h + 4 * margin_y + header_text_h + legend_h

#     canvas = Image.new("RGB", (table_w, table_h), "white")
#     draw = ImageDraw.Draw(canvas)

#     # Headers and images
#     for i, name in enumerate(common_ordered):
#         label = desired_labels[i] if i < len(desired_labels) else name

#         x = margin_x + i * w
#         bbox = draw.textbbox((0, 0), label, font=font)
#         text_w = bbox[2] - bbox[0]
#         draw.text((x + (w - text_w) // 2, margin_y // 2), label, fill="black", font=font)

#         avg_y = margin_y + header_text_h
#         mae_y = avg_y + h + margin_y

#         canvas.paste(avg_imgs[name], (x, avg_y))
#         canvas.paste(mae_imgs[name], (x, mae_y))

#     row_font = get_scalable_font(int(h * 0.15))
#     draw.text((20, margin_y + header_text_h + h // 2 - 20), "Mean", fill="black", font=row_font)
#     draw.text((20, margin_y + header_text_h + h + margin_y + h // 2 - 20), "MAE", fill="black", font=row_font)

#     # Legend (fixed 4 bins)
#     colors = get_fixed_colors_rgb()
#     labels = ["<1.00", "1.00–2.00", "2.00–3.00", ">3.00"]

#     n = len(colors)
#     seg_w = (table_w - 2 * margin_x) // n
#     legend_y = table_h - legend_h - margin_y

#     for i, (color, label_txt) in enumerate(zip(colors, labels)):
#         x0 = margin_x + i * seg_w
#         x1 = x0 + seg_w

#         draw.rectangle(
#             [x0, legend_y, x1, legend_y + legend_h // 2],
#             fill=tuple(color.tolist()),
#             outline="black"
#         )

#         bbox = draw.textbbox((0, 0), label_txt, font=font)
#         tw = bbox[2] - bbox[0]
#         draw.text(
#             (x0 + (seg_w - tw) // 2, legend_y + legend_h // 2 + 5),
#             label_txt,
#             fill="black",
#             font=font
#         )

#     canvas.save(output_path)
#     print(f"✅ Saved combined table with custom order/labels: {output_path}")


# # ===== MAIN RUN =====
# if __name__ == "__main__":
#     mesh_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/ERN/properly trimmed-2dl/SNH/SNH NEW/for heatmaps"
#     desktop_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/"
#     output_folder = os.path.join(desktop_folder, "heatmaps-20260226-mean-mae")
#     output_csv = os.path.join(mesh_folder, "Hausdorff_metrics_meshlab.csv")

#     process_mesh_folder(mesh_folder, output_folder, output_csv)
#     generate_screenshots_from_ply(output_folder, only_avg=False)
#     combine_avg_mae_table(
#         output_folder,
#         output_path=os.path.join(output_folder, "combined_avg_mae_table.png")
#     )

##########################################################################################################################################


# import os
# import re
# import numpy as np
# import pandas as pd
# import trimesh
# from PIL import Image, ImageDraw, ImageFont
# from pathlib import Path
# import pyvista as pv
# from trimesh.transformations import rotation_matrix
# import pymeshlab as ml  # PyMeshLab for Hausdorff distance

# # Enable off-screen rendering for PyVista
# os.environ["PYVISTA_OFF_SCREEN"] = "true"
# pv.start_xvfb()

# # ===== FONT & DRAWING UTILITIES =====
# def get_scalable_font(font_size: int):
#     font_paths = [
#         "/mnt/c/Windows/Fonts/arial.ttf",
#         "arial.ttf",
#         "C:/Windows/Fonts/arial.ttf",
#         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
#         "./DejaVuSans-Bold.ttf",
#     ]
#     for path in font_paths:
#         if Path(path).exists():
#             print(f"🔤 Using font: {path}")
#             return ImageFont.truetype(path, size=font_size)
#     print("⚠️ No scalable font found. Using fallback bitmap font.")
#     return ImageFont.load_default()


# # ===== FIXED COLOR SCALE (4 bins: green/yellow/orange/red) =====
# def get_fixed_bins_mm():
#     # Internal edges for np.digitize -> 4 bins:
#     # idx 0: <1, idx 1: [1,2), idx 2: [2,3), idx 3: >=3
#     return np.array([1.0, 2.0, 3.0], dtype=float)


# def get_fixed_colors_rgb():
#     # Order corresponds to bins: <1, 1–2, 2–3, >=3
#     return np.array([
#         [0, 200, 0],      # green
#         [255, 255, 0],    # yellow
#         [255, 165, 0],    # orange
#         [255, 0, 0],      # red
#     ], dtype=np.uint8)


# # ===== HAUSDORFF COMPUTATION =====
# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path):
#     ms = ml.MeshSet()
#     ms.load_new_mesh(ref_mesh_path)     # mesh 0
#     ms.load_new_mesh(target_mesh_path)  # mesh 1

#     # Optional absolute cutoff to prevent crazy values blowing things up
#     ABS_CUTOFF_MM = 154.6744
#     maxdist_arg = ml.PureValue(ABS_CUTOFF_MM) if hasattr(ml, "PureValue") else None

#     vcount = ms.mesh(1).vertex_number()
#     kwargs = dict(
#         sampledmesh=1,
#         targetmesh=0,
#         savesample=True,
#         samplevert=True,
#         sampleedge=False,
#         sampleface=False,
#         samplenum=max(vcount, 1)
#     )
#     if maxdist_arg is not None:
#         kwargs["maxdist"] = maxdist_arg

#     try:
#         ms.apply_filter("get_hausdorff_distance", **kwargs)
#     except TypeError:
#         # Some builds don't accept maxdist
#         kwargs.pop("maxdist", None)
#         ms.apply_filter("get_hausdorff_distance", **kwargs)

#     # Hausdorff sample mesh is appended at the end
#     new_ids = list(range(ms.number_meshes()))
#     sample_dists = np.asarray(ms.mesh(new_ids[-1]).vertex_scalar_array(), dtype=float)

#     if sample_dists.size:
#         min_mm = float(sample_dists.min())
#         max_mm = float(sample_dists.max())
#         mean_mm = float(sample_dists.mean())
#         rms_mm = float(np.sqrt((sample_dists ** 2).mean()))
#     else:
#         min_mm = max_mm = mean_mm = rms_mm = 0.0

#     # Per-vertex distance on mesh(1) from ref mesh(0)
#     ms.apply_filter(
#         "compute_scalar_by_distance_from_another_mesh_per_vertex",
#         measuremesh=1,
#         refmesh=0,
#         signeddist=False
#     )
#     per_vertex_dists = np.asarray(ms.mesh(1).vertex_scalar_array(), dtype=float)

#     metrics = {
#         "Reference Mesh": os.path.basename(ref_mesh_path),
#         "Target Mesh": os.path.basename(target_mesh_path),
#         "Minimum Distance (mm)": min_mm,
#         "Maximum Distance (mm)": max_mm,
#         "Mean Distance (mm)": mean_mm,
#         "RMS Distance (mm)": rms_mm,
#         "Reference Vertex Count": ms.mesh(0).vertex_number()
#     }
#     return per_vertex_dists, metrics


# # ===== HEATMAP EXPORT =====
# def save_colored_mesh_ply(target_mesh_path, hausdorff_distances, out_path):
#     mesh = trimesh.load_mesh(target_mesh_path, process=False)

#     # Face distance = mean of the vertex distances on that face
#     face_distances = np.mean(hausdorff_distances[mesh.faces], axis=1)

#     bins = get_fixed_bins_mm()
#     color_map = get_fixed_colors_rgb()

#     # 0 for <1, 1 for [1,2), 2 for [2,3), 3 for >=3
#     bin_indices = np.digitize(face_distances, bins, right=False)
#     bin_indices = np.clip(bin_indices, 0, len(color_map) - 1)

#     face_colors = color_map[bin_indices]

#     # Explode faces so each triangle can have a flat colour
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


# # ===== METRICS CSV =====
# def save_metrics_to_csv(metrics_list, output_csv):
#     df = pd.DataFrame(metrics_list)
#     df.reset_index(drop=True, inplace=True)
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Saved metrics: {output_csv}")


# # ===== UTILS =====
# def clean_short_name(filename, max_len=15):
#     base = os.path.splitext(filename)[0]
#     base = re.sub(r"\W+", "", base)
#     return base[:max_len]


# # ===== MAIN PIPELINE =====
# def process_mesh_folder(mesh_folder, output_folder, output_csv):
#     print(f"🟢 Processing folder: {mesh_folder}")
#     os.makedirs(output_folder, exist_ok=True)

#     mesh_files = sorted([f for f in os.listdir(mesh_folder)
#                          if f.lower().endswith((".ply", ".stl", ".obj"))])
#     mesh_paths = {f: os.path.join(mesh_folder, f) for f in mesh_files}

#     all_metrics = []
#     per_vertex_dists = {clean_short_name(f): [] for f in mesh_files}

#     for ref_name in mesh_files:
#         for tgt_name in mesh_files:
#             if ref_name == tgt_name:
#                 continue

#             ref_path, tgt_path = mesh_paths[ref_name], mesh_paths[tgt_name]
#             print(f"🔹 {ref_name} vs {tgt_name}")

#             hausdorff_distances, metrics = compute_hausdorff_metrics(ref_path, tgt_path)
#             all_metrics.append(metrics)

#             tgt_base = clean_short_name(tgt_name)
#             per_vertex_dists[tgt_base].append(hausdorff_distances)

#             ref_base = clean_short_name(ref_name)
#             out_name = f"heatmap_{ref_base}_vs_{tgt_base}.ply"
#             out_path = os.path.join(output_folder, out_name)
#             save_colored_mesh_ply(tgt_path, hausdorff_distances, out_path)

#     print("\n📊 Averaging vertex-wise Hausdorff distances per mesh...")
#     for tgt_name, dist_list in per_vertex_dists.items():
#         if not dist_list:
#             continue

#         stack = np.stack(dist_list, axis=0)
#         mean_dists = np.mean(stack, axis=0)
#         mae_dists = np.mean(np.abs(stack), axis=0)

#         tgt_file = next(f for f in mesh_files if clean_short_name(f) == tgt_name)
#         tgt_path = mesh_paths[tgt_file]

#         save_colored_mesh_ply(
#             tgt_path, mean_dists,
#             os.path.join(output_folder, f"avg_heatmap_{tgt_name}.ply")
#         )
#         save_colored_mesh_ply(
#             tgt_path, mae_dists,
#             os.path.join(output_folder, f"mae_heatmap_{tgt_name}.ply")
#         )
#         print(f"✅ Saved mean & MAE heatmaps for {tgt_name}")

#     save_metrics_to_csv(all_metrics, output_csv)


# # ===== SCREENSHOTS =====
# def generate_screenshots_from_ply(ply_folder, only_avg=True):
#     pitch_deg, yaw_deg, roll_deg = 50, 0, 135

#     for filename in os.listdir(ply_folder):
#         if not filename.lower().endswith(".ply"):
#             continue
#         if only_avg and not (filename.startswith("avg_heatmap_") or filename.startswith("mae_heatmap_")):
#             continue

#         try:
#             ply_path = os.path.join(ply_folder, filename)
#             mesh = trimesh.load(ply_path, process=False)

#             rotation_center = mesh.centroid
#             R_final = (
#                 rotation_matrix(np.radians(roll_deg), [0, 0, 1]) @
#                 rotation_matrix(np.radians(yaw_deg), [0, 1, 0]) @
#                 rotation_matrix(np.radians(pitch_deg), [1, 0, 0])
#             )
#             T_pre = np.eye(4)
#             T_post = np.eye(4)
#             T_pre[:3, 3] = -rotation_center
#             T_post[:3, 3] = rotation_center
#             mesh.apply_transform(T_post @ R_final @ T_pre)

#             pv_mesh = pv.wrap(mesh)

#             if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
#                 pv_mesh.cell_data["colors"] = mesh.visual.face_colors[:, :3]
#                 scalars_name = "colors"
#                 rgb = True
#                 use_cell = True
#             elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
#                 pv_mesh.point_data["colors"] = mesh.visual.vertex_colors[:, :3]
#                 scalars_name = "colors"
#                 rgb = True
#                 use_cell = False
#             else:
#                 scalars_name = None
#                 rgb = False
#                 use_cell = False

#             plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
#             plotter.set_background("white")

#             if scalars_name is None:
#                 plotter.add_mesh(pv_mesh, show_scalar_bar=False, backface_culling=True)
#             else:
#                 plotter.add_mesh(
#                     pv_mesh,
#                     scalars=scalars_name,
#                     rgb=rgb,
#                     show_scalar_bar=False,
#                     backface_culling=True,
#                     preference="cell" if use_cell else "point"
#                 )

#             plotter.view_vector((1, 1, 1))
#             screenshot_path = os.path.join(ply_folder, f"{os.path.splitext(filename)[0]}.png")
#             plotter.screenshot(screenshot_path)
#             plotter.close()
#             print(f"📸 Saved screenshot: {screenshot_path}")

#         except Exception as e:
#             print(f"❌ Failed to render {filename}: {e}")


# # ===== COMBINE TABLE (MAE ONLY, NO ROW LABEL, IMAGES 10% BIGGER) =====
# def combine_avg_mae_table(
#     screenshot_folder,
#     output_path="combined_table.png"
# ):
#     """
#     - Uses ONLY the MAE heatmaps in the combined image (no Mean row).
#     - Removes the 'MAE' row label entirely.
#     - Scales each pasted heatmap image up by 10%.
#     - Keeps your custom column ordering + custom labels.
#     """
#     mae_imgs = {}

#     for fname in os.listdir(screenshot_folder):
#         fpath = os.path.join(screenshot_folder, fname)
#         if fname.startswith("mae_heatmap_") and fname.lower().endswith(".png"):
#             key = re.sub(r"^mae_heatmap_|\.png$", "", fname)
#             mae_imgs[key] = Image.open(fpath)

#     if not mae_imgs:
#         print("❌ No MAE heatmap PNGs found.")
#         return

#     common_all = sorted(mae_imgs.keys())

#     # Your desired order/labels
#     desired_order = ["emc", "gos", "rad", "sah", "ukt", "cha", "nck"]
#     desired_labels = ["A_3dMD", "B_3dMD", "C_3dMD", "D_3dMD", "E_Vectra", "F_Vectra", "G_Artec"]

#     # Match by prefix, but allow reuse (so nck can appear multiple times if desired)
#     def match_key_for_code(code: str):
#         code_l = code.lower()
#         for k in common_all:
#             if k.lower().startswith(code_l):
#                 return k
#         return None

#     common_ordered = []
#     for code in desired_order:
#         k = match_key_for_code(code)
#         if k is None:
#             print(f"⚠️ Missing MAE image for code '{code}' (no key starting with '{code}').")
#             continue
#         common_ordered.append(k)

#     if not common_ordered:
#         print("❌ None of the requested ordered codes were found as MAE images.")
#         return

#     # Base image size + scale up by 10%
#     base_w, base_h = next(iter(mae_imgs.values())).size
#     scale = 1.10
#     w = int(round(base_w * scale))
#     h = int(round(base_h * scale))

#     # Fonts + margins (header based on scaled image height)
#     font = get_scalable_font(int(h * 0.12))
#     margin_x, margin_y, legend_h = 120, 60, 90
#     header_text_h = int(h * 0.25)

#     # Single-row table (MAE only)
#     table_w = len(common_ordered) * w + margin_x
#     table_h = h + 3 * margin_y + header_text_h + legend_h

#     canvas = Image.new("RGB", (table_w, table_h), "white")
#     draw = ImageDraw.Draw(canvas)

#     # Paste MAE images only (scaled), with headers
#     for i, name in enumerate(common_ordered):
#         label = desired_labels[i] if i < len(desired_labels) else name

#         x = margin_x + i * w

#         bbox = draw.textbbox((0, 0), label, font=font)
#         text_w = bbox[2] - bbox[0]
#         draw.text((x + (w - text_w) // 2, margin_y // 2), label, fill="black", font=font)

#         y = margin_y + header_text_h

#         # scale image 10% bigger
#         img = mae_imgs[name].resize((w, h), resample=Image.Resampling.LANCZOS)
#         canvas.paste(img, (x, y))

#     # Legend (fixed 4 bins)
#     colors = get_fixed_colors_rgb()
#     labels = ["<1.00", "1.00–2.00", "2.00–3.00", ">3.00"]

#     n = len(colors)
#     seg_w = (table_w - 2 * margin_x) // n
#     legend_y = table_h - legend_h - margin_y

#     for i, (color, label_txt) in enumerate(zip(colors, labels)):
#         x0 = margin_x + i * seg_w
#         x1 = x0 + seg_w

#         draw.rectangle(
#             [x0, legend_y, x1, legend_y + legend_h // 2],
#             fill=tuple(color.tolist()),
#             outline="black"
#         )

#         bbox = draw.textbbox((0, 0), label_txt, font=font)
#         tw = bbox[2] - bbox[0]
#         draw.text(
#             (x0 + (seg_w - tw) // 2, legend_y + legend_h // 2 + 5),
#             label_txt,
#             fill="black",
#             font=font
#         )

#     canvas.save(output_path)
#     print(f"✅ Saved combined MAE-only table (10% bigger, no row label): {output_path}")


# # ===== MAIN RUN =====
# if __name__ == "__main__":
#     mesh_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/ERN/properly trimmed-2dl/SNH/SNH NEW/for heatmaps"
#     desktop_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/"
#     output_folder = os.path.join(desktop_folder, "heatmaps-20260226-mean-mae")
#     output_csv = os.path.join(mesh_folder, "Hausdorff_metrics_meshlab.csv")

#     process_mesh_folder(mesh_folder, output_folder, output_csv)
#     generate_screenshots_from_ply(output_folder, only_avg=False)

#     # MAE-only combined output
#     combine_avg_mae_table(
#         output_folder,
#         output_path=os.path.join(output_folder, "combined_mae_table.png")
#     )

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
def get_scalable_font(font_size: int):
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


# ===== FIXED COLOR SCALE (4 bins: green/yellow/orange/red) =====
def get_fixed_bins_mm():
    return np.array([1.0, 2.0, 3.0], dtype=float)


def get_fixed_colors_rgb():
    return np.array([
        [0, 200, 0],      # green
        [255, 255, 0],    # yellow
        [255, 165, 0],    # orange
        [255, 0, 0],      # red
    ], dtype=np.uint8)


# ===== HAUSDORFF COMPUTATION =====
def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(ref_mesh_path)     # mesh 0
    ms.load_new_mesh(target_mesh_path)  # mesh 1

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
        kwargs.pop("maxdist", None)
        ms.apply_filter("get_hausdorff_distance", **kwargs)

    new_ids = list(range(ms.number_meshes()))
    sample_dists = np.asarray(ms.mesh(new_ids[-1]).vertex_scalar_array(), dtype=float)

    if sample_dists.size:
        min_mm = float(sample_dists.min())
        max_mm = float(sample_dists.max())
        mean_mm = float(sample_dists.mean())
        rms_mm = float(np.sqrt((sample_dists ** 2).mean()))
    else:
        min_mm = max_mm = mean_mm = rms_mm = 0.0

    ms.apply_filter(
        "compute_scalar_by_distance_from_another_mesh_per_vertex",
        measuremesh=1,
        refmesh=0,
        signeddist=False
    )
    per_vertex_dists = np.asarray(ms.mesh(1).vertex_scalar_array(), dtype=float)

    metrics = {
        "Reference Mesh": os.path.basename(ref_mesh_path),
        "Target Mesh": os.path.basename(target_mesh_path),
        "Minimum Distance (mm)": min_mm,
        "Maximum Distance (mm)": max_mm,
        "Mean Distance (mm)": mean_mm,
        "RMS Distance (mm)": rms_mm,
        "Reference Vertex Count": ms.mesh(0).vertex_number()
    }
    return per_vertex_dists, metrics


# ===== HEATMAP EXPORT =====
def save_colored_mesh_ply(target_mesh_path, hausdorff_distances, out_path):
    mesh = trimesh.load_mesh(target_mesh_path, process=False)

    face_distances = np.mean(hausdorff_distances[mesh.faces], axis=1)

    bins = get_fixed_bins_mm()
    color_map = get_fixed_colors_rgb()

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
def process_mesh_folder(mesh_folder, output_folder, output_csv):
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

            hausdorff_distances, metrics = compute_hausdorff_metrics(ref_path, tgt_path)
            all_metrics.append(metrics)

            tgt_base = clean_short_name(tgt_name)
            per_vertex_dists[tgt_base].append(hausdorff_distances)

            ref_base = clean_short_name(ref_name)
            out_name = f"heatmap_{ref_base}_vs_{tgt_base}.ply"
            out_path = os.path.join(output_folder, out_name)
            save_colored_mesh_ply(tgt_path, hausdorff_distances, out_path)

    print("\n📊 Averaging vertex-wise Hausdorff distances per mesh...")
    for tgt_name, dist_list in per_vertex_dists.items():
        if not dist_list:
            continue

        stack = np.stack(dist_list, axis=0)
        mean_dists = np.mean(stack, axis=0)
        mae_dists = np.mean(np.abs(stack), axis=0)

        tgt_file = next(f for f in mesh_files if clean_short_name(f) == tgt_name)
        tgt_path = mesh_paths[tgt_file]

        save_colored_mesh_ply(
            tgt_path, mean_dists,
            os.path.join(output_folder, f"avg_heatmap_{tgt_name}.ply")
        )
        save_colored_mesh_ply(
            tgt_path, mae_dists,
            os.path.join(output_folder, f"mae_heatmap_{tgt_name}.ply")
        )
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
                scalars_name = "colors"
                rgb = True
                use_cell = True
            elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                pv_mesh.point_data["colors"] = mesh.visual.vertex_colors[:, :3]
                scalars_name = "colors"
                rgb = True
                use_cell = False
            else:
                scalars_name = None
                rgb = False
                use_cell = False

            plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
            plotter.set_background("white")

            if scalars_name is None:
                plotter.add_mesh(pv_mesh, show_scalar_bar=False, backface_culling=True)
            else:
                plotter.add_mesh(
                    pv_mesh,
                    scalars=scalars_name,
                    rgb=rgb,
                    show_scalar_bar=False,
                    backface_culling=True,
                    preference="cell" if use_cell else "point"
                )

            plotter.view_vector((1, 1, 1))
            screenshot_path = os.path.join(ply_folder, f"{os.path.splitext(filename)[0]}.png")
            plotter.screenshot(screenshot_path)
            plotter.close()
            print(f"📸 Saved screenshot: {screenshot_path}")

        except Exception as e:
            print(f"❌ Failed to render {filename}: {e}")


# ===== AUTO-CROP WHITE BORDERS (to make skulls larger) =====
def autocrop_white(img: Image.Image, threshold: int = 240, pad: int = 6) -> Image.Image:
    """
    Crops away near-white borders. Assumes white background.
    threshold: higher -> stricter definition of 'white' (240-250 usually good)
    pad: keep a few pixels of padding after crop
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img)
    nonwhite = np.any(arr < threshold, axis=2)

    if not np.any(nonwhite):
        return img

    ys, xs = np.where(nonwhite)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)

    return img.crop((x0, y0, x1 + 1, y1 + 1))


def resize_to_fit(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """
    Resize to fit inside (box_w, box_h) preserving aspect ratio,
    then center on a white canvas of exactly (box_w, box_h).
    """
    img = img.convert("RGB")
    iw, ih = img.size
    s = min(box_w / iw, box_h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))

    img_r = img.resize((nw, nh), resample=Image.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), "white")
    ox = (box_w - nw) // 2
    oy = (box_h - nh) // 2
    canvas.paste(img_r, (ox, oy))
    return canvas


# ===== COMBINE TABLE (MAE ONLY, NO ROW LABEL, CROPPED CONTENT) =====
def combine_avg_mae_table(
    screenshot_folder,
    output_path="combined_table.png"
):
    """
    - Uses ONLY the MAE heatmaps in the combined image (no Mean row).
    - Removes the 'MAE' row label entirely.
    - Crops white borders so the skulls appear larger.
    - Keeps your custom column ordering + custom labels.
    """
    mae_imgs = {}

    for fname in os.listdir(screenshot_folder):
        fpath = os.path.join(screenshot_folder, fname)
        if fname.startswith("mae_heatmap_") and fname.lower().endswith(".png"):
            key = re.sub(r"^mae_heatmap_|\.png$", "", fname)
            mae_imgs[key] = Image.open(fpath)

    if not mae_imgs:
        print("❌ No MAE heatmap PNGs found.")
        return

    common_all = sorted(mae_imgs.keys())

    # Your desired order/labels
    desired_order = ["emc", "gos", "rad", "sah", "ukt", "cha", "nck"]
    desired_labels = ["A_3dMD", "B_3dMD", "C_3dMD", "D_3dMD", "E_Vectra", "F_Vectra", "G_Artec"]

    def match_key_for_code(code: str):
        code_l = code.lower()
        for k in common_all:
            if k.lower().startswith(code_l):
                return k
        return None

    common_ordered = []
    for code in desired_order:
        k = match_key_for_code(code)
        if k is None:
            print(f"⚠️ Missing MAE image for code '{code}' (no key starting with '{code}').")
            continue
        common_ordered.append(k)

    if not common_ordered:
        print("❌ None of the requested ordered codes were found as MAE images.")
        return

    # Table cell size:
    # Keep the original PNG size for each cell, but skulls get bigger via cropping.
    base_w, base_h = next(iter(mae_imgs.values())).size
    w, h = base_w, base_h

    # Fonts + margins
    font = get_scalable_font(int(h * 0.12))
    margin_x, margin_y, legend_h = 120, 60, 90
    header_text_h = int(h * 0.25)

    # Single-row table (MAE only)
    table_w = len(common_ordered) * w + margin_x
    table_h = h + 3 * margin_y + header_text_h + legend_h

    canvas = Image.new("RGB", (table_w, table_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Paste MAE images only (cropped -> fitted), with headers
    for i, name in enumerate(common_ordered):
        label = desired_labels[i] if i < len(desired_labels) else name
        x = margin_x + i * w

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (w - text_w) // 2, margin_y // 2), label, fill="black", font=font)

        y = margin_y + header_text_h

        img0 = mae_imgs[name].convert("RGB")
        img1 = autocrop_white(img0, threshold=240, pad=6)  # TUNE: threshold 235-250, pad 0-12
        img2 = resize_to_fit(img1, w, h)
        canvas.paste(img2, (x, y))

    # Legend (fixed 4 bins)
    colors = get_fixed_colors_rgb()
    legend_labels = ["<1.00", "1.00–2.00", "2.00–3.00", ">3.00"]

    n = len(colors)
    seg_w = (table_w - 2 * margin_x) // n
    legend_y = table_h - legend_h - margin_y

    for i, (color, label_txt) in enumerate(zip(colors, legend_labels)):
        x0 = margin_x + i * seg_w
        x1 = x0 + seg_w

        draw.rectangle(
            [x0, legend_y, x1, legend_y + legend_h // 2],
            fill=tuple(color.tolist()),
            outline="black"
        )

        bbox = draw.textbbox((0, 0), label_txt, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (x0 + (seg_w - tw) // 2, legend_y + legend_h // 2 + 5),
            label_txt,
            fill="black",
            font=font
        )

    canvas.save(output_path)
    print(f"✅ Saved combined MAE-only table (cropped skulls): {output_path}")


# ===== MAIN RUN =====
if __name__ == "__main__":
    mesh_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/ERN/properly trimmed-2dl/MHT/MHT NEW/for heatmaps"
    desktop_folder = "/mnt/c/Users/klay.luke.PSYDUCK/Desktop/"
    output_folder = os.path.join(desktop_folder, "MHT-heatmaps-20260226-mean-mae")
    output_csv = os.path.join(mesh_folder, "Hausdorff_metrics_meshlab.csv")

    process_mesh_folder(mesh_folder, output_folder, output_csv)
    generate_screenshots_from_ply(output_folder, only_avg=False)

    # MAE-only combined output
    combine_avg_mae_table(
        output_folder,
        output_path=os.path.join(output_folder, "combined_mae_table.png")
    )