# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import os

# def generate_scaled_bins(max_distance, num_bins=8):
#     """Generates scaled color bins based on the max distance, keeping evenly spaced intervals."""
#     return np.linspace(0, max_distance, num_bins + 1)  # num_bins+1 to create boundaries

# def generate_scaled_colors():
#     """Returns an 8-bin RGB color gradient from Green to Red."""
#     return np.array([
#         [0.2549, 1.0000, 0.0000],  # Green
#         [0.6275, 1.0000, 0.0000],  # Light Green
#         [0.9804, 1.0000, 0.0000],  # Yellow-Green
#         [1.0000, 0.6706, 0.0000],  # Yellow
#         [1.0000, 0.2784, 0.0000],  # Light Orange
#         [1.0000, 0.0000, 0.0000],  # Red-Orange
#         [0.8000, 0.0000, 0.0000],  # Dark Red
#         [0.6000, 0.0000, 0.0000]   # Deep Red (Added 8th color)
#     ])

# def generate_colorbar(threshold, save_path="colorbar.png"):
#     """Generates and saves a horizontal color bar image based on the threshold value."""
    
#     # Define bins and colors
#     color_bins = generate_scaled_bins(threshold)
#     colors = generate_scaled_colors()
    
#     # Ensure color list matches bin count
#     if len(colors) < len(color_bins) - 1:
#         raise ValueError(f"Not enough colors ({len(colors)}) for {len(color_bins)-1} bins!")

#     # Create colormap
#     cmap = mcolors.ListedColormap(colors)
#     norm = mcolors.BoundaryNorm(color_bins, cmap.N)

#     # Create figure
#     fig, ax = plt.subplots(figsize=(6, 1))
#     cb = plt.colorbar(
#         plt.cm.ScalarMappable(cmap=cmap, norm=norm),
#         cax=ax, orientation="horizontal"
#     )

#     cb.set_label("Hausdorff Distance (mm)")
#     cb.set_ticks(color_bins)
#     cb.ax.set_xticklabels([f"{x:.1f}" for x in color_bins])

#     # Save the figure
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
    
#     # Print the saved file location
#     abs_path = os.path.abspath(save_path)
#     print(f"Color bar saved at: {abs_path}")

# if __name__ == "__main__":
#     threshold_value = 10  # Set the maximum threshold distance
#     save_path = "/mnt/c/Users/User/Desktop/2025 ERN/2025 ERN/SNH/craniumpy/colorbar.png"  # Ensure it's a file path, not a folder

#     generate_colorbar(threshold_value, save_path)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def generate_scaled_bins(max_distance, num_bins=8):
    """Generates scaled color bins based on the max distance, keeping evenly spaced intervals."""
    return np.linspace(0, max_distance, num_bins + 1)  # num_bins+1 to create boundaries

def generate_scaled_colors():
    """Returns an 8-bin RGB color gradient from Green to Red."""
    return np.array([
        [0.2549, 1.0000, 0.0000],  # Green
        [0.6275, 1.0000, 0.0000],  # Light Green
        [0.9804, 1.0000, 0.0000],  # Yellow-Green
        [1.0000, 0.6706, 0.0000],  # Yellow
        [1.0000, 0.2784, 0.0000],  # Light Orange
        [1.0000, 0.0000, 0.0000],  # Red-Orange
        [0.8000, 0.0000, 0.0000],  # Dark Red
        [0.6000, 0.0000, 0.0000]   # Deep Red
    ])

def generate_colorbar(threshold, save_path, orientation="horizontal"):
    """Generates and saves a color bar image based on the threshold value."""
    
    # Define bins and colors
    color_bins = generate_scaled_bins(threshold)
    colors = generate_scaled_colors()
    
    # Ensure color list matches bin count
    if len(colors) < len(color_bins) - 1:
        raise ValueError(f"Not enough colors ({len(colors)}) for {len(color_bins)-1} bins!")

    # Create colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(color_bins, cmap.N)

    # Set figure size based on orientation
    if orientation == "horizontal":
        figsize = (6, 1)
    elif orientation == "vertical":
        figsize = (1, 6)
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=ax, orientation=orientation
    )

    cb.set_label("Hausdorff Distance (mm)")
    cb.set_ticks(color_bins)
    cb.ax.set_yticklabels([f"{x:.1f}" for x in color_bins]) if orientation == "vertical" else cb.ax.set_xticklabels([f"{x:.1f}" for x in color_bins])

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Print the saved file location
    abs_path = os.path.abspath(save_path)
    print(f"Color bar saved at: {abs_path}")

if __name__ == "__main__":
    threshold_value = 5.0  # Set the maximum threshold distance
    save_dir = "/mnt/c/Users/User/Desktop/2025 ERN/2025 ERN/SNH/"  # Base directory

    # Save horizontal and vertical color bars
    generate_colorbar(threshold_value, os.path.join(save_dir, f"{threshold_value}_mm_colourbar_horizontal.png"), "horizontal")
    generate_colorbar(threshold_value, os.path.join(save_dir, f"{threshold_value}_mm_colourbar_vertical.png"), "vertical")
