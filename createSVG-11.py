import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Import for loading the background image
import matplotlib as mpl
import requests
import json

def maidenhead_to_latlon(grid_square: str) -> tuple[float, float]:
    """
    Convert a Maidenhead Grid Square to latitude/longitude coordinates.
    Returns the coordinates of the center of the grid square.
    
    Args:
        grid_square (str): Maidenhead Grid Square (e.g., "FN31pr")
        
    Returns:
        tuple[float, float]: (latitude, longitude) pair
        
    Example:
        >>> maidenhead_to_latlon("FN31pr")
        (41.375, -72.875)
    """
    grid_square = grid_square.upper().strip()
    
    if not (4 <= len(grid_square) <= 6 and len(grid_square) % 2 == 0):
        raise ValueError("Grid square must be 4 or 6 characters long")
        
    # Field (1st pair)
    lon = (ord(grid_square[0]) - ord('A')) * 20 - 180
    lat = (ord(grid_square[1]) - ord('A')) * 10 - 90
    
    # Square (2nd pair)
    lon += (ord(grid_square[2]) - ord('0')) * 2
    lat += (ord(grid_square[3]) - ord('0'))
    
    # Subsquare (3rd pair, if present)
    if len(grid_square) == 6:
        lon += (ord(grid_square[4]) - ord('A')) / 12
        lat += (ord(grid_square[5]) - ord('A')) / 24
        # Move to center of subsquare
        lon += 1/24
        lat += 1/48
    else:
        # Move to center of square
        lon += 1
        lat += 0.5
        
    return (lat, lon)

def create_bucket_id(lat: float, lon: float, delta_lat: float = 4, delta_lon: float = 6) -> int:
   """
   Create a bucket ID from latitude and longitude coordinates.
   
   Args:
       lat (float): Latitude
       lon (float): Longitude
       delta_lat (float): Latitude step size (default: 4)
       delta_lon (float): Longitude step size (default: 6)
       
   Returns:
       int: Bucket ID
   """
   MIN_LAT = -90
   MIN_LON = -180
   
   return 1000 * int((lat - MIN_LAT) / delta_lat) + int((lon - MIN_LON) / delta_lon)

def process_dxview_url(base_url: str) -> int:
   """
   Process a DXView perspective URL to get the bucket ID.
   
   Args:
       base_url (str): Base URL in format "https://hf.dxview.org/perspective/{GRID_SQUARE}"
       
   Returns:
       int: Bucket ID
       
   Example:
       >>> process_dxview_url("https://hf.dxview.org/perspective/KM38lr")
       45123  # example ID
   """
   # Extract grid square from URL
   try:
       grid_square = base_url.split("/perspective/")[1]
   except IndexError:
       raise ValueError("Invalid URL format. Expected format: https://hf.dxview.org/perspective/{GRID_SQUARE}")
   
   # Convert grid square to lat/lon
   try:
       lat, lon = maidenhead_to_latlon(grid_square)
   except ValueError as e:
       raise ValueError(f"Invalid grid square in URL: {str(e)}")
   
   # Return just the bucket ID
   return create_bucket_id(lat, lon)

def fetch_zones(band: int, bucket_id: int) -> dict:
   """
   Fetch zone data from DXView for a specific band and bucket ID.
   
   Args:
       band (int): Band in MHz - must be one of: 1,3,5,7,10,14,18,21,24,28,50
       bucket_id (int): Bucket ID for the location
       
   Returns:
       dict: Parsed JSON response from the server
       
   Raises:
       ValueError: If band is invalid
       requests.RequestException: If network request fails
       json.JSONDecodeError: If response is not valid JSON
   """
   valid_bands = {1, 3, 5, 7, 10, 14, 18, 21, 24, 28, 50}
   
   if band not in valid_bands:
       raise ValueError(f"Band must be one of: {sorted(valid_bands)}")
       
   url = f"https://hf.dxview.org/map/refresh?dx=1&band={band}&id={bucket_id}"
   
   try:
       response = requests.get(url)
       response.raise_for_status()  # Raises exception for 4XX/5XX status codes
       return response.json()
       
   except requests.RequestException as e:
       raise requests.RequestException(f"Failed to fetch data: {str(e)}")
   except json.JSONDecodeError as e:
       raise json.JSONDecodeError(f"Failed to parse response as JSON: {str(e)}", e.doc, e.pos)   

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
    scaled_x = x * scale_y + off_y # Scale X-axis
    scaled_y = y * scale_x + off_x # Scale Y-axis
    print([x,y,'scaled',scaled_y, scaled_x])
    return scaled_y, scaled_x  # Rotate 90 degrees counterclockwise    

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
    circle_radius = 5
    for zone in data["zones"]:
        for signal_type in colors:
            if signal_type in zone:
                paths = zone[signal_type]
                if isinstance(paths[0][0], list):  # Nested case
                    for sub_path in paths:
                        # Rotate and collect coordinates
                        rotated_coords = [
                            #rotate_90_counterclockwise(point[0] * 200 + 400, point[1] * -200 + 300)
                            rotate_90_counterclockwise(point[0], point[1])
                            for point in sub_path
                        ]
                        if len(rotated_coords) == 1:  # Isolated data point
                            x, y = rotated_coords[0]
                            circle = plt.Circle((x, y), circle_radius, linewidth=None, color=colors[signal_type], alpha=0.6, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                            ax.add_patch(circle)
                            all_x.append(x)
                            all_y.append(y)
                        elif len(rotated_coords) == 2:  # Two-point line case
                            smoothed_polygon = create_smoothed_thickened_polygon(
                                rotated_coords, thickness=20, num_points=100
                            )
                            if smoothed_polygon:
                                x, y = smoothed_polygon.exterior.xy
                                #y[0] = y[0]+50
                                #y[1] = y[1]+50
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
                        #rotate_90_counterclockwise(point[0] * 200 + 400, point[1] * -200 + 300)
                        rotate_90_counterclockwise(point[0], point[1])
                        for point in paths
                    ]
                    if len(rotated_coords) == 1:  # Isolated data point
                        x, y = rotated_coords[0]
                        circle = plt.Circle((x, y), circle_radius, linewidth=None, color=colors[signal_type], alpha=0.6, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                        ax.add_patch(circle)
                        all_x.append(x)
                        all_y.append(y)
                    elif len(rotated_coords) == 2:  # Two-point line case
                        smoothed_polygon = create_smoothed_thickened_polygon(
                            rotated_coords, thickness=20, num_points=100
                        )
                        if smoothed_polygon:
                            x, y = smoothed_polygon.exterior.xy
                            #y[0] = y[0]-50
                            #y[1] = y[1]-50
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
    #ax.set_xlim(min(all_x) - 50, max(all_x) + 50)
    #ax.set_ylim(min(all_y) - 50, max(all_y) + 50)

# Example data and colors (replace 'data' with your actual JSON input)
colors = {
    "cw_perimeter": "#CC00CC",
    "digital_perimeter": "#00CCCC",
    "ssb_perimeter": "#CCCC00"
}

scale_x = 159  # 180%
scale_y = 201  # 118%
off_x = +470
off_y= +200
perspective = "JM68hn"
band = 28

dxview_url = f"https://hf.dxview.org/perspective/{perspective}"
data = fetch_zones(band, process_dxview_url(dxview_url))


#data = {"zones":[{"id":"6084379_0","band":28,"is_muf":False,"digital_perimeter":[[-0.576,1.99]]},{"id":"6089372_0","band":28,"is_muf":False,"cw_perimeter":[[[-0.456,0.487]]],"digital_perimeter":[[-0.456,0.487],[-0.445,0.539]]},{"id":"6089578_0","band":28,"is_muf":False,"cw_perimeter":[[[-0.443,-0.859]]]},{"id":"6089889_0","band":28,"is_muf":False,"cw_perimeter":[[[-0.309,3.097]]]},{"id":"6085953_0","band":28,"is_muf":False,"cw_perimeter":[[[0.23,1.353]]]},{"id":"6090261_0","band":28,"is_muf":False,"ssb_perimeter":[[[0.384,0.785],[0.384,0.89],[0.419,1.047],[0.454,0.995],[0.454,0.89],[0.559,0.628],[0.663,0.785],[0.698,0.838],[0.768,0.838],[0.803,0.785],[0.768,0.733],[0.698,0.733],[0.593,0.576],[0.559,0.524],[0.524,0.576],[0.419,0.733]]]},{"id":"6090156_0","band":28,"is_muf":False,"ssb_perimeter":[[[0.62,1.971]]],"cw_perimeter":[[[0.454,2.147],[0.489,2.199],[0.524,2.251],[0.559,2.304],[0.628,2.304],[0.663,2.251],[0.698,2.094],[0.733,2.042],[0.698,1.99],[0.663,1.937],[0.628,1.885],[0.593,1.937],[0.489,2.094]]],"digital_perimeter":[[0.384,1.937],[0.419,1.99],[0.454,2.147],[0.489,2.199],[0.524,2.251],[0.559,2.304],[0.593,2.356],[0.628,2.409],[0.663,2.356],[0.663,2.251],[0.698,2.094],[0.733,2.042],[0.698,1.99],[0.663,1.937],[0.628,1.885],[0.593,1.937],[0.559,1.99],[0.454,1.937],[0.419,1.885]]},{"id":"6090261_1","band":28,"is_muf":False,"ssb_perimeter":[[[0.663,-0.488]],[[0.663,-0.052],[0.663,0.052],[0.698,0.105],[0.803,0.262],[0.838,0.314],[0.977,0.314],[1.082,0.262],[1.117,0.209],[1.152,0.157],[1.117,0.105],[0.977,0.105],[0.942,-0.052],[0.908,-0.105],[0.838,-0.105],[0.698,-0.105]],[[0.992,0.93]]],"cw_perimeter":[[[0.454,-0.262],[0.489,-0.209],[0.559,-0.209],[0.593,-0.157],[0.628,-0.105],[0.663,0.052],[0.663,0.262],[0.698,0.314],[0.838,0.314],[0.873,0.367],[0.908,0.419],[0.942,0.471],[0.942,0.681],[0.977,0.942],[1.012,0.89],[1.012,0.681],[1.047,0.628],[1.082,0.576],[1.082,0.471],[1.117,0.209],[1.152,0.157],[1.117,0.105],[1.047,0.105],[1.012,-0.052],[1.012,-0.157],[0.977,-0.209],[0.942,-0.157],[0.838,-0.105],[0.803,-0.157],[0.768,-0.209],[0.628,-0.209],[0.593,-0.262],[0.628,-0.419],[0.663,-0.471],[0.628,-0.524],[0.593,-0.471],[0.489,-0.314]]],"digital_perimeter":[[0.454,-0.262],[0.489,-0.209],[0.559,-0.209],[0.593,-0.052],[0.628,0.0],[0.663,0.052],[0.663,0.262],[0.698,0.314],[0.803,0.367],[0.838,0.419],[0.908,0.419],[0.942,0.471],[0.942,0.681],[0.977,0.942],[1.012,0.89],[1.012,0.681],[1.047,0.628],[1.082,0.576],[1.082,0.471],[1.117,0.209],[1.152,0.157],[1.117,0.105],[1.082,0.052],[1.047,0.0],[1.012,-0.157],[0.977,-0.209],[0.942,-0.157],[0.838,-0.105],[0.803,-0.157],[0.768,-0.209],[0.698,-0.209],[0.593,-0.262],[0.628,-0.419],[0.663,-0.471],[0.628,-0.524],[0.593,-0.471],[0.489,-0.314]]},{"id":"6090190_0","band":28,"is_muf":False,"digital_perimeter":[[0.742,-1.241],[0.735,-1.238]]},{"id":"6090261_2","band":28,"is_muf":False,"cw_perimeter":[[[0.907,1.221]]]},{"id":"6090261_3","band":28,"is_muf":False,"digital_perimeter":[[0.936,1.524]]},{"id":"6090026_0","band":28,"is_muf":False,"digital_perimeter":[[0.913,1.82]]}]}
#data = {"zones":[{"id":"6096508_0","band":7,"is_muf":False,"ssb_perimeter":[[[-0.351,1.007],[-0.344,1.108]]]},{"id":"6096491_0","band":7,"is_muf":False,"ssb_perimeter":[[[0.227,1.353]]]},{"id":"6095172_0","band":7,"is_muf":False,"ssb_perimeter":[[[0.524,-2.042],[0.559,-1.99],[0.593,-1.833],[0.559,-1.78],[0.524,-1.728],[0.524,-1.518],[0.559,-1.361],[0.628,-1.361],[0.663,-1.204],[0.698,-1.152],[0.768,-1.152],[0.803,-1.204],[0.803,-1.309],[0.768,-1.676],[0.698,-1.676],[0.663,-1.833],[0.698,-1.885],[0.733,-1.937],[0.838,-2.094],[0.873,-1.937],[0.908,-1.885],[0.942,-1.937],[0.942,-2.042],[0.908,-2.094],[0.873,-2.147],[0.838,-2.199],[0.803,-2.147],[0.698,-1.99],[0.628,-1.99],[0.593,-2.042],[0.559,-2.094]]],"cw_perimeter":[[[0.384,-1.414],[0.419,-1.361],[0.489,-1.361],[0.663,-1.204],[0.698,-1.152],[0.768,-1.152],[0.803,-1.204],[0.803,-1.309],[0.768,-1.676],[0.698,-1.676],[0.663,-1.833],[0.698,-1.885],[0.733,-1.937],[0.838,-2.094],[0.873,-1.937],[0.908,-1.885],[0.942,-1.937],[0.942,-2.042],[0.908,-2.094],[0.873,-2.147],[0.838,-2.199],[0.803,-2.147],[0.698,-1.99],[0.628,-1.99],[0.593,-2.042],[0.559,-2.094],[0.524,-2.042],[0.559,-1.99],[0.593,-1.833],[0.559,-1.78],[0.524,-1.728],[0.524,-1.518],[0.419,-1.466]]]},{"id":"6096736_0","band":7,"is_muf":False,"ssb_perimeter":[[[0.441,0.894],[0.441,0.969]]]},{"id":"6096736_1","band":7,"is_muf":False,"ssb_perimeter":[[[0.524,0.576],[0.559,0.628],[0.663,0.785],[0.698,0.838],[0.838,0.838],[0.873,0.89],[0.908,0.942],[0.942,0.995],[0.977,1.047],[1.012,0.995],[1.012,0.89],[1.047,0.524],[1.082,0.471],[1.117,0.419],[1.152,0.367],[1.152,0.262],[1.117,0.105],[1.082,0.052],[1.047,0.0],[1.012,-0.052],[0.977,-0.105],[0.908,-0.105],[0.733,-0.157],[0.698,-0.209],[0.663,-0.157],[0.663,-0.052],[0.698,0.0],[0.768,0.0],[0.803,0.157],[0.768,0.314],[0.733,0.367],[0.733,0.471],[0.698,0.524],[0.559,0.524]]],"cw_perimeter":[[[0.524,0.576],[0.559,0.628],[0.663,0.785],[0.698,0.838],[0.838,0.838],[0.873,0.89],[0.908,0.942],[0.942,0.995],[0.977,1.047],[1.012,0.995],[1.012,0.89],[1.047,0.524],[1.082,0.471],[1.117,0.419],[1.152,0.367],[1.152,0.262],[1.117,0.105],[1.082,0.052],[1.047,0.0],[1.012,-0.052],[0.977,-0.105],[0.908,-0.105],[0.803,-0.157],[0.768,-0.209],[0.698,-0.209],[0.663,-0.157],[0.663,-0.052],[0.698,0.0],[0.733,0.052],[0.768,0.105],[0.803,0.157],[0.768,0.314],[0.733,0.367],[0.733,0.471],[0.698,0.524],[0.559,0.524]]]},{"id":"6096736_2","band":7,"is_muf":False,"ssb_perimeter":[[[0.98,1.618],[0.873,1.846],[0.92,1.807]]],"cw_perimeter":[[[0.873,1.518],[0.908,1.571],[0.942,1.623],[0.908,1.78],[0.873,1.833],[0.908,1.885],[0.942,1.833],[0.977,1.676],[1.012,1.623],[0.977,1.571],[0.942,1.518],[0.908,1.466]]]},{"id":"6096736_3","band":7,"is_muf":False,"ssb_perimeter":[[[1.119,-0.383]]]}]}
# Plot setup for isolated points with circles
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
ax.set_title("Map with Circles (Radius 0.1) for Isolated Points")
ax.set_xlabel("Longitude (rotated)")
ax.set_ylabel("Latitude (rotated)")

# Load the background image
background_image = mpimg.imread("Background-Template2w.png")  # Load the PNG file

#ax.imshow(background_image, origin='upper', aspect='equal', extent=[-500,2500,150,900])  # Set background with scaling
ax.imshow(background_image, origin='upper', aspect='equal', extent=[0,1927,0,662])  # Set background with scaling


# Draw interpolated and smoothed paths, with circles for isolated points
draw_interpolated_smoothed_paths_with_circles(data, colors, ax)

# Add legend
ax.legend(loc='upper right')
ax.patch.set_alpha(0.0)
plt.show()