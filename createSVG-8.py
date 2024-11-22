import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Import for loading the background image

# Function to densify paths by adding intermediate points
def densify_path(coords, interval=10):
    if len(coords) < 2:  # Skip invalid geometries
        return coords
    line = LineString(coords)
    num_points = int(line.length / interval) + 1
    return [line.interpolate(float(i) / num_points, normalized=True).coords[0] for i in range(num_points + 1)]

# Adjusted interpolate_path to ensure sufficient points
def interpolate_path(coords, num_points=100):
    if len(coords) < 4:  # Ensure sufficient points for interpolation
        coords = densify_path(coords, interval=10)  # Add intermediate points
    if len(coords) < 4:  # Double-check after densification
        return np.array(coords)  # Skip interpolation if still insufficient
    coords = np.array(coords)
    tck, _ = splprep([coords[:, 0], coords[:, 1]], s=0, per=True)
    new_coords = splev(np.linspace(0, 1, num_points), tck)
    return np.column_stack(new_coords)

# Function to rotate points by 90 degrees counterclockwise
def rotate_90_counterclockwise(x, y):
    # Scale the x and y values before rotation
    scaled_x = x * (1 + scale_x)
    scaled_y = y * (1 + scale_y)
    return -scaled_y, scaled_x  # Rotate 90 degrees counterclockwise    

# Function to create a smoothed thickened polygon for lines
def create_smoothed_thickened_polygon(coords, thickness=20, num_points=100):
    if len(coords) != 2:  # Ensure it's a two-point line
        return None
    line = LineString(coords)
    thickened_polygon = line.buffer(thickness, cap_style=2, join_style=2)  # Create a thickened polygon
    if not thickened_polygon.is_empty:
        # Apply interpolation for smoothing
        smoothed_coords = interpolate_path(list(thickened_polygon.exterior.coords), num_points=num_points)
        return Polygon(smoothed_coords)
    return None

# Function to handle isolated data points and plot them as circles with radius 0.1
def draw_interpolated_smoothed_paths_with_circles(data, colors, ax):
    all_x = []
    all_y = []

    for zone in data["zones"]:
        for signal_type in colors:
            if signal_type in zone:
                paths = zone[signal_type]
                if isinstance(paths[0][0], list):  # Nested case
                    for sub_path in paths:
                        # Rotate and collect coordinates
                        rotated_coords = [
                            rotate_90_counterclockwise(point[0] * 200 + 400, point[1] * -200 + 300)
                            for point in sub_path
                        ]
                        if len(rotated_coords) == 1:  # Isolated data point
                            x, y = rotated_coords[0]
                            circle = plt.Circle((x, y), 20, linewidth=None, color=colors[signal_type], alpha=0.6, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                            ax.add_patch(circle)
                            all_x.append(x)
                            all_y.append(y)
                        elif len(rotated_coords) == 2:  # Two-point line case
                            smoothed_polygon = create_smoothed_thickened_polygon(
                                rotated_coords, thickness=20, num_points=100
                            )
                            if smoothed_polygon:
                                x, y = smoothed_polygon.exterior.xy
                                ax.fill(x, y, linewidth=None, color=colors[signal_type], alpha=0.4, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                                all_x.extend(x)
                                all_y.extend(y)
                        else:
                            # Close the polygon if not already closed
                            if rotated_coords[0] != rotated_coords[-1]:
                                rotated_coords.append(rotated_coords[0])
                            # Interpolate to ensure smoothness
                            interpolated_coords = interpolate_path(rotated_coords, num_points=100)
                            x, y = interpolated_coords[:, 0], interpolated_coords[:, 1]
                            ax.fill(x, y, linewidth=None, color=colors[signal_type], alpha=0.4, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                            all_x.extend(x)
                            all_y.extend(y)
                else:  # Single path case or isolated point
                    rotated_coords = [
                        rotate_90_counterclockwise(point[0] * 200 + 400, point[1] * -200 + 300)
                        for point in paths
                    ]
                    if len(rotated_coords) == 1:  # Isolated data point
                        x, y = rotated_coords[0]
                        circle = plt.Circle((x, y), 10, linewidth=None, color=colors[signal_type], alpha=0.6, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                        ax.add_patch(circle)
                        all_x.append(x)
                        all_y.append(y)
                    elif len(rotated_coords) == 2:  # Two-point line case
                        smoothed_polygon = create_smoothed_thickened_polygon(
                            rotated_coords, thickness=20, num_points=100
                        )
                        if smoothed_polygon:
                            x, y = smoothed_polygon.exterior.xy
                            ax.fill(x, y, linewidth=None, color=colors[signal_type], alpha=0.4, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                            all_x.extend(x)
                            all_y.extend(y)
                    elif len(rotated_coords) >= 3:  # Polygon or multi-point case
                        if rotated_coords[0] != rotated_coords[-1]:
                            rotated_coords.append(rotated_coords[0])
                        interpolated_coords = interpolate_path(rotated_coords, num_points=100)
                        x, y = interpolated_coords[:, 0], interpolated_coords[:, 1]
                        ax.fill(x, y, linewidth=None, color=colors[signal_type], alpha=0.4, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                        all_x.extend(x)
                        all_y.extend(y)

    # Adjust axis limits
    xlim = min(all_x) - 50, max(all_x) + 50
    ylim = min(all_y) - 50, max(all_y) + 50
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Example data and colors (replace 'data' with your actual JSON input)
colors = {
    "cw_perimeter": "#CCCCCC",
    "digital_perimeter": "#CCCCCC",
    "ssb_perimeter": "#CCCCCC"
}

scale_x = 1.2  # 180%
scale_y = 1.18  # 118%

data = {"zones":[{"id":"6084379_0","band":28,"is_muf":False,"digital_perimeter":[[-0.576,1.99]]},{"id":"6089372_0","band":28,"is_muf":False,"cw_perimeter":[[[-0.456,0.487]]],"digital_perimeter":[[-0.456,0.487],[-0.445,0.539]]},{"id":"6089578_0","band":28,"is_muf":False,"cw_perimeter":[[[-0.443,-0.859]]]},{"id":"6089889_0","band":28,"is_muf":False,"cw_perimeter":[[[-0.309,3.097]]]},{"id":"6085953_0","band":28,"is_muf":False,"cw_perimeter":[[[0.23,1.353]]]},{"id":"6090261_0","band":28,"is_muf":False,"ssb_perimeter":[[[0.384,0.785],[0.384,0.89],[0.419,1.047],[0.454,0.995],[0.454,0.89],[0.559,0.628],[0.663,0.785],[0.698,0.838],[0.768,0.838],[0.803,0.785],[0.768,0.733],[0.698,0.733],[0.593,0.576],[0.559,0.524],[0.524,0.576],[0.419,0.733]]]},{"id":"6090156_0","band":28,"is_muf":False,"ssb_perimeter":[[[0.62,1.971]]],"cw_perimeter":[[[0.454,2.147],[0.489,2.199],[0.524,2.251],[0.559,2.304],[0.628,2.304],[0.663,2.251],[0.698,2.094],[0.733,2.042],[0.698,1.99],[0.663,1.937],[0.628,1.885],[0.593,1.937],[0.489,2.094]]],"digital_perimeter":[[0.384,1.937],[0.419,1.99],[0.454,2.147],[0.489,2.199],[0.524,2.251],[0.559,2.304],[0.593,2.356],[0.628,2.409],[0.663,2.356],[0.663,2.251],[0.698,2.094],[0.733,2.042],[0.698,1.99],[0.663,1.937],[0.628,1.885],[0.593,1.937],[0.559,1.99],[0.454,1.937],[0.419,1.885]]},{"id":"6090261_1","band":28,"is_muf":False,"ssb_perimeter":[[[0.663,-0.488]],[[0.663,-0.052],[0.663,0.052],[0.698,0.105],[0.803,0.262],[0.838,0.314],[0.977,0.314],[1.082,0.262],[1.117,0.209],[1.152,0.157],[1.117,0.105],[0.977,0.105],[0.942,-0.052],[0.908,-0.105],[0.838,-0.105],[0.698,-0.105]],[[0.992,0.93]]],"cw_perimeter":[[[0.454,-0.262],[0.489,-0.209],[0.559,-0.209],[0.593,-0.157],[0.628,-0.105],[0.663,0.052],[0.663,0.262],[0.698,0.314],[0.838,0.314],[0.873,0.367],[0.908,0.419],[0.942,0.471],[0.942,0.681],[0.977,0.942],[1.012,0.89],[1.012,0.681],[1.047,0.628],[1.082,0.576],[1.082,0.471],[1.117,0.209],[1.152,0.157],[1.117,0.105],[1.047,0.105],[1.012,-0.052],[1.012,-0.157],[0.977,-0.209],[0.942,-0.157],[0.838,-0.105],[0.803,-0.157],[0.768,-0.209],[0.628,-0.209],[0.593,-0.262],[0.628,-0.419],[0.663,-0.471],[0.628,-0.524],[0.593,-0.471],[0.489,-0.314]]],"digital_perimeter":[[0.454,-0.262],[0.489,-0.209],[0.559,-0.209],[0.593,-0.052],[0.628,0.0],[0.663,0.052],[0.663,0.262],[0.698,0.314],[0.803,0.367],[0.838,0.419],[0.908,0.419],[0.942,0.471],[0.942,0.681],[0.977,0.942],[1.012,0.89],[1.012,0.681],[1.047,0.628],[1.082,0.576],[1.082,0.471],[1.117,0.209],[1.152,0.157],[1.117,0.105],[1.082,0.052],[1.047,0.0],[1.012,-0.157],[0.977,-0.209],[0.942,-0.157],[0.838,-0.105],[0.803,-0.157],[0.768,-0.209],[0.698,-0.209],[0.593,-0.262],[0.628,-0.419],[0.663,-0.471],[0.628,-0.524],[0.593,-0.471],[0.489,-0.314]]},{"id":"6090190_0","band":28,"is_muf":False,"digital_perimeter":[[0.742,-1.241],[0.735,-1.238]]},{"id":"6090261_2","band":28,"is_muf":False,"cw_perimeter":[[[0.907,1.221]]]},{"id":"6090261_3","band":28,"is_muf":False,"digital_perimeter":[[0.936,1.524]]},{"id":"6090026_0","band":28,"is_muf":False,"digital_perimeter":[[0.913,1.82]]}]}
# Plot setup for isolated points with circles
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
ax.set_title("Map with Circles (Radius 0.1) for Isolated Points")
ax.set_xlabel("Longitude (rotated)")
ax.set_ylabel("Latitude (rotated)")

# Draw interpolated and smoothed paths, with circles for isolated points
draw_interpolated_smoothed_paths_with_circles(data, colors, ax)

# Load the background image
background_image = mpimg.imread("Background-Template.png")  # Load the PNG file

pos = ax.get_position()
pos.x0=0
pos.y0=0
pos.x1=1
pos.y1=1

imgax = fig.add_axes(pos,     # new axes with same position
    label='image',                          # label to ensure imgaxes is different from a
    zorder=-1,                              # put image below the plot
    xticks=[], yticks=[])                   # remove the ticks
imgax.axis('off')
imgax.imshow(background_image, origin='upper',  aspect='equal')  # Set background with scaling

# Add legend
ax.legend(loc='upper right')
ax.patch.set_alpha(0.0)
plt.show()
