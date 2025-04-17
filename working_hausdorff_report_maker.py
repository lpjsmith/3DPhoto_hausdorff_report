# import os
# import numpy as np
# import pyvista as pv
# import trimesh
# import vtk
# import pandas as pd
# from scipy.spatial import cKDTree
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib import colors
# from reportlab.platypus import Table, TableStyle

# def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path):
#     """Computes Hausdorff distances using KDTree and returns statistical metrics."""

#     # Load meshes with Trimesh
#     ref_mesh = trimesh.load_mesh(ref_mesh_path, process=False)
#     target_mesh = trimesh.load_mesh(target_mesh_path, process=False)

#     # Convert vertices to NumPy arrays
#     ref_points = ref_mesh.vertices
#     target_points = target_mesh.vertices

#     # 🔹 Use KDTree to efficiently find nearest distances
#     kdtree = cKDTree(ref_points)
#     hausdorff_distances, _ = kdtree.query(target_points)

#     # 🔹 Compute Statistical Metrics
#     min_distance = np.min(hausdorff_distances)
#     max_distance = np.max(hausdorff_distances)
#     mean_distance = np.mean(hausdorff_distances)
#     std_dev_distance = np.std(hausdorff_distances)
#     rms_distance = np.sqrt(np.mean(hausdorff_distances ** 2))
#     percentile_95 = np.percentile(hausdorff_distances, 95)
#     under_2mm_count = np.sum(hausdorff_distances < 2.0)

#     metrics = {
#         "Minimum Distance (mm)": min_distance,
#         "Maximum Distance (mm)": max_distance,
#         "Mean Distance (mm)": mean_distance,
#         "Standard Deviation (mm)": std_dev_distance,
#         "RMS Distance (mm)": rms_distance,
#         "95th Percentile Distance (mm)": percentile_95,
#         "Points < 2mm": under_2mm_count,
#     }

#     return hausdorff_distances, metrics

# def save_metrics_to_csv(metrics, output_csv):
#     """Saves computed metrics to a CSV file."""
#     df = pd.DataFrame([metrics])
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Metrics saved to {output_csv}")

# def generate_report(ref_mesh_path, target_mesh_path, output_pdf, output_csv):
#     """Generates a PDF report comparing two STL files and saves computed metrics."""

#     print(f"Processing: {target_mesh_path} vs {ref_mesh_path}")

#     # Compute Hausdorff metrics
#     hausdorff_distances, metrics = compute_hausdorff_metrics(ref_mesh_path, target_mesh_path)

#     # Save metrics to CSV
#     save_metrics_to_csv(metrics, output_csv)

#     # Define output screenshot paths
#     folder_path = os.path.dirname(target_mesh_path)
#     front_view_path = os.path.join(folder_path, 'front_view.png')
#     left_view_path = os.path.join(folder_path, 'left_view.png')
#     top_view_path = os.path.join(folder_path, 'top_view.png')
#     isometric_view_path = os.path.join(folder_path, 'isometric_view.png')

#     # Initialize PDF canvas
#     c = canvas.Canvas(output_pdf, pagesize=letter)
#     c.setFont("Helvetica", 12)

#     # Add metric values to the PDF
#     c.drawString(50, 750, f"Reference Mesh: {os.path.basename(ref_mesh_path)}")
#     c.drawString(50, 730, f"Target Mesh: {os.path.basename(target_mesh_path)}")

#     y_offset = 700
#     for key, value in metrics.items():
#         c.drawString(50, y_offset, f"{key}: {value:.3f} mm")
#         y_offset -= 20

#     # Add table with task placeholders
#     task_table_data = [['TASK', 'DETAILS', 'SIGNOFF', 'DATE'],
#                        ['Task 1', '', '', ''],
#                        ['Task 2', '', '', ''],
#                        ['Task 3', '', '', ''],
#                        ['Task 4', '', '', ''],
#                        ['Task 5', '', '', '']]

#     task_table = Table(task_table_data, colWidths=[100, 200, 100, 100])
#     task_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#                                     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#                                     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#                                     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#                                     ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#                                     ('BACKGROUND', (0, 1), (-1, -1), colors.beige)]))

#     # Add the table to the PDF
#     task_table.wrapOn(c, 0, 0)
#     task_table.drawOn(c, 50, 150)

#     # Save PDF
#     c.save()
#     print(f"📄 PDF Report saved: {output_pdf}")

# if __name__ == "__main__":
#     # Define reference and target STL files
#     ref_mesh_path = r"/mnt/c/Users/User/Desktop/123/MHT_trimmed/trimmed - posterior block removed/registered global ICP meshlab/EMC_AC_MHT_20240702.000016_trim_meshlab_reg.stl"
#     target_mesh_path = r"/mnt/c/Users/User/Desktop/123/MHT_trimmed/trimmed - posterior block removed/registered global ICP meshlab/GOSH_AC_MHT_20240708.000184_trim_meshlab_reg.stl"
#     output_pdf = r"/mnt/c/Users/User/Desktop/123/MHT_trimmed/trimmed - posterior block removed/registered global ICP meshlab/report.pdf"
#     output_csv = r"/mnt/c/Users/User/Desktop/123/MHT_trimmed/trimmed - posterior block removed/registered global ICP meshlab/metrics.csv"

#     generate_report(ref_mesh_path, target_mesh_path, output_pdf, output_csv)


## above this is one mesh at a time below is a folder of meshes pair compare

import os
import numpy as np
import pyvista as pv
import trimesh
import pandas as pd
import itertools
from scipy.spatial import cKDTree
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from PIL import Image

def compute_hausdorff_metrics(ref_mesh_path, target_mesh_path):
    """Computes Hausdorff distances using KDTree and returns statistical metrics."""

    # Load meshes with Trimesh
    ref_mesh = trimesh.load_mesh(ref_mesh_path, process=False)
    target_mesh = trimesh.load_mesh(target_mesh_path, process=False)

    # Convert vertices to NumPy arrays
    ref_points = ref_mesh.vertices
    target_points = target_mesh.vertices

    # 🔹 Use KDTree to efficiently find nearest distances
    kdtree = cKDTree(ref_points)
    hausdorff_distances, _ = kdtree.query(target_points)

    # 🔹 Compute Statistical Metrics
    min_distance = np.min(hausdorff_distances)
    max_distance = np.max(hausdorff_distances)
    mean_distance = np.mean(hausdorff_distances)
    std_dev_distance = np.std(hausdorff_distances)
    rms_distance = np.sqrt(np.mean(hausdorff_distances ** 2))
    percentile_95 = np.percentile(hausdorff_distances, 95)
    under_2mm_count = np.sum(hausdorff_distances < 2.0)

    metrics = {
        "Minimum Distance (mm)": min_distance,
        "Maximum Distance (mm)": max_distance,
        "Mean Distance (mm)": mean_distance,
        "Standard Deviation (mm)": std_dev_distance,
        "RMS Distance (mm)": rms_distance,
        "95th Percentile Distance (mm)": percentile_95,
        "Points < 2mm": under_2mm_count,
    }

    return hausdorff_distances, metrics

def save_metrics_to_csv(metrics, output_csv):
    """Saves computed Hausdorff metrics to a CSV file."""
    df = pd.DataFrame(metrics)
    df.to_csv(output_csv, index=False)
    print(f"✅ Metrics successfully saved to {output_csv}")
    
def load_existing_heatmaps(heatmap_folder):
    """Scans the heatmaps folder and builds a list of mesh names from PNG filenames."""
    if not os.path.exists(heatmap_folder):
        print("❌ No heatmap folder found. Ensure PNGs are in 'heatmaps/' before running this script.")
        return [], {}

    heatmap_files = sorted([f for f in os.listdir(heatmap_folder) if f.endswith(".png")])
    heatmap_images = {}
    mesh_names = set()

    # Extract reference and target mesh names from filenames
    for filename in heatmap_files:
        if "heatmap_" in filename and "_vs_" in filename:
            parts = filename.replace("heatmap_", "").replace(".png", "").split("_vs_")
            if len(parts) == 2:
                ref_mesh, target_mesh = parts
                full_path = os.path.join(heatmap_folder, filename)
                heatmap_images[(ref_mesh, target_mesh)] = full_path
                mesh_names.add(ref_mesh)
                mesh_names.add(target_mesh)

    # Sort mesh names alphabetically
    mesh_names = sorted(mesh_names)

    return mesh_names, heatmap_images

def generate_matrix_pdf(mesh_files, heatmap_images, output_pdf):
    """Generates a PDF with a matrix of heatmap images from existing PNG files."""
    c = canvas.Canvas(output_pdf, pagesize=landscape(letter))
    c.setFont("Helvetica", 10)

    cell_size = 80  # 🔹 Slightly reduced to fit more images
    left_margin = 50  # Space for row labels
    bottom_margin = 0  # 🔹 Corrected to properly position images from the bottom
    text_offset = 20

    num_meshes = len(mesh_files)

    # 🔹 Auto-scale images if too many meshes exist
    if num_meshes > 10:
        cell_size = max(60, 900 // (num_meshes + 1))  # Dynamically reduce cell size

    # 🔹 Draw column headers (Truncated Mesh Names)
    for col, mesh_name in enumerate(mesh_files):
        short_name = mesh_name[:7]  # 🔹 Use only the first 7 characters
        c.drawString(left_margin + (col + 1) * cell_size, bottom_margin + (num_meshes + 1) * cell_size, short_name)

    # 🔹 Draw row headers (Truncated Mesh Names) & Heatmap Matrix
    for row, ref_mesh in enumerate(mesh_files):
        short_name = ref_mesh[:7]  # 🔹 Use only the first 7 characters
        c.drawString(left_margin - text_offset, bottom_margin + (num_meshes - row) * cell_size, short_name)

        for col, target_mesh in enumerate(mesh_files):
            if ref_mesh == target_mesh:
                continue  # Skip self-comparison

            img_path = heatmap_images.get((ref_mesh, target_mesh))
            if img_path and os.path.exists(img_path):
                c.drawImage(
                    img_path,
                    left_margin + (col + 1) * cell_size,
                    bottom_margin + (num_meshes - row) * cell_size,
                    width=cell_size,
                    height=cell_size
                )
            else:
                print(f"⚠️ Missing heatmap for {ref_mesh} vs {target_mesh}")

    c.save()
    print(f"📄 PDF Heatmap Matrix saved: {output_pdf}")

def generate_scaled_bins(max_distance, num_bins=8):
    """Generates scaled color bins based on the max distance, keeping evenly spaced intervals."""
    return np.linspace(0, max_distance, num_bins)

def generate_scaled_colors():
    """Returns the 8-bin RGB color gradient from Green to Red."""
    return np.array([
        [0.2549, 1.0000, 0.0000],  # Green
        [0.6275, 1.0000, 0.0000],  # Light Green
        [0.9804, 1.0000, 0.0000],  # Yellow-Green
        [1.0000, 0.6706, 0.0000],  # Yellow
        [1.0000, 0.2784, 0.0000],  # Light Orange
        [1.0000, 0.0000, 0.0000],  # Red-Orange
        [1.0000, 0.0000, 0.0000]   # Full Red
    ])

def compute_hausdorff_heatmap(ref_mesh_path, target_mesh_path):
    """Computes Hausdorff distances and applies a heatmap with 8 bins."""
    
    # Compute Hausdorff distances
    hausdorff_distances, _ = compute_hausdorff_metrics(ref_mesh_path, target_mesh_path)

    # Load target mesh into PyVista
    target_mesh = trimesh.load_mesh(target_mesh_path, process=False)
    pv_mesh = pv.wrap(target_mesh)

    # Compute per-face distances
    face_distances = np.mean(hausdorff_distances[target_mesh.faces], axis=1)

    # 🔹 Generate dynamic bins based on `max_distance`
    max_distance = 2.0
    color_bins = generate_scaled_bins(max_distance)
    color_map = generate_scaled_colors()

    # Convert to RGBA (Ensure full opacity)
    color_map_rgba = np.hstack([color_map, np.ones((color_map.shape[0], 1))]) * 255

    # Assign discrete colors per-face
    face_colors = np.zeros((len(face_distances), 4), dtype=np.uint8)

    for i, d in enumerate(face_distances):
        for j in range(len(color_bins) - 1):
            if color_bins[j] <= d < color_bins[j + 1]:
                face_colors[i] = color_map_rgba[j]
                break
        if d >= 2.0000:
            face_colors[i] = [255, 0, 0, 255]  # Ensure anything ≥ 2mm is full red

    # Apply colors
    pv_mesh.cell_data["Heatmap"] = face_colors

    return pv_mesh

# def render_heatmap(mesh, img_path):
#     """Renders the heatmap and saves the image."""
#     plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
#     mesh.cell_data["Heatmap"][:, 3] = 255  # Ensure full opacity
#     plotter.set_background("white")
#     plotter.add_mesh(mesh, scalars="Heatmap", rgb=True, show_scalar_bar=False)
#     plotter.view_xy()
#     plotter.screenshot(img_path, transparent_background=False)
#     plotter.close()

def render_heatmap(mesh, img_path, crop_factor=0.8):
    """Renders the heatmap, center crops it, and saves the image."""
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    mesh.cell_data["Heatmap"][:, 3] = 255  # Ensure full opacity
    plotter.set_background("white")
    plotter.add_mesh(mesh, scalars="Heatmap", rgb=True, show_scalar_bar=False)
    plotter.view_xy()
    
    temp_img_path = img_path.replace(".png", "_temp.png")
    plotter.screenshot(temp_img_path, transparent_background=False)
    plotter.close()

    # Load image and apply center crop
    img = Image.open(temp_img_path)
    width, height = img.size
    new_size = int(min(width, height) * crop_factor)

    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size

    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(img_path)

    # Cleanup temp file
    os.remove(temp_img_path)

def process_mesh_folder(mesh_folder, output_folder, output_pdf, output_csv):
    """Runs the full pipeline: Computes Hausdorff metrics, then generates a matrix PDF."""
    print(f"🟢 Processing meshes in folder: {mesh_folder}")
    os.makedirs(output_folder, exist_ok=True)

    mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.lower().endswith(('.stl', '.obj'))])
    mesh_paths = {f: os.path.join(mesh_folder, f) for f in mesh_files}

    all_metrics = []

    # Compute Hausdorff Metrics for each pair
    for ref_mesh_name, target_mesh_name in itertools.product(mesh_files, repeat=2):
        if ref_mesh_name == target_mesh_name:
            continue  # Skip self-comparison

        ref_mesh_path = mesh_paths[ref_mesh_name]
        target_mesh_path = mesh_paths[target_mesh_name]

        print(f"🔹 Computing metrics for: {ref_mesh_name} vs {target_mesh_name}")
        metrics = compute_hausdorff_metrics(ref_mesh_path, target_mesh_path)
        all_metrics.append(metrics)

        mesh = compute_hausdorff_heatmap(ref_mesh_path, target_mesh_path)

        # Save heatmap image
        img_filename = f"heatmap_{ref_mesh_name[:7]}_vs_{target_mesh_name[:7]}.png"
        img_path = os.path.join(output_folder, img_filename)
        render_heatmap(mesh, img_path)

    # Save Metrics to CSV
    save_metrics_to_csv(all_metrics, output_csv)

    # Load existing heatmaps & generate PDF matrix
    heatmap_folder = os.path.join(mesh_folder, "heatmaps")
    mesh_names, heatmap_images = load_existing_heatmaps(heatmap_folder)

    generate_matrix_pdf(mesh_names, heatmap_images, output_pdf)

if __name__ == "__main__":
    mesh_folder = '/mnt/c/Users/User/Desktop/2025 ERN/2025 ERN/MHT/craniumpy'
    output_folder = os.path.join(mesh_folder, "heatmaps")
    output_pdf = os.path.join(mesh_folder, "heatmap_matrix.pdf")
    output_csv = os.path.join(mesh_folder, "metrics.csv")

    # 🔹 Full Processing: Compute Hausdorff Metrics + Generate Report
    #process_mesh_folder(mesh_folder, output_folder, output_pdf, output_csv)

    # 🔹 Use below if you just want to rebuild the PDF without recalculating metrics
    process_heatmap_folder(output_folder, output_pdf)

