import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
                            circle = plt.Circle((x, y+50), 3, linewidth=None, color=colors[signal_type], alpha=0.6, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
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
                        circle = plt.Circle((x, y+50), 10, linewidth=None, color=colors[signal_type], alpha=0.6, label=signal_type if not ax.get_legend_handles_labels()[1].count(signal_type) else "")
                        ax.add_patch(circle)
                        all_x.append(x)
                        all_y.append(y)
                    elif len(rotated_coords) == 2:  # Two-point line case
                        smoothed_polygon = create_smoothed_thickened_polygon(
                            rotated_coords, thickness=20, num_points=100
                        )
                        if smoothed_polygon:
                            x, y = smoothed_polygon.exterior.xy
                            y[0] = y[0]-50
                            y[1] = y[1]-50
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

def create_plots(data, colors, dxview_url, zoom_level=3):
    # Get user coordinates from grid square
    grid_square = dxview_url.split("/perspective/")[1]
    user_lat, user_lon = maidenhead_to_latlon(grid_square)
    print(f"User coordinates: lat={user_lat}, lon={user_lon}")
    
    # Create main figure
    fig = plt.figure(figsize=(10, 6))
    
    # Create main plot
    main_ax = fig.add_axes([0, 0, 1, 1])
    main_ax.set_aspect('equal')
    
    # Load and display background image for main plot
    background_image = mpimg.imread("Background-Template-5.png")
    main_ax.imshow(background_image, origin='upper', aspect='equal', extent=[0, 1927, 0, 662])
    
    # Draw data on main plot
    draw_interpolated_smoothed_paths_with_circles(data, colors, main_ax)
    
    # Calculate zoomed plot position and dimensions
    bbox = main_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    main_width = bbox.width * fig.dpi
    main_height = bbox.height * fig.dpi
    
    zoom_width_px = main_width * 0.5
    zoom_height_px = main_height * 0.8
    
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    zoom_x = 510 / fig_width
    zoom_y = 50 / fig_height
    zoom_width = zoom_width_px / fig_width
    zoom_height = zoom_height_px / fig_height
    
    # Create zoomed inset plot
    zoom_ax = fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height], zorder=2)
    zoom_ax.set_aspect('equal')
    
    # Calculate zoom window in lat/lon coordinates
    lat_range = 180 / zoom_level  # Full range is -90 to 90 (180 degrees)
    lon_range = 360 / zoom_level  # Full range is -180 to 180 (360 degrees)
    
    # Calculate lat/lon bounds for zoom window
    lat_min = max(-90, user_lat - lat_range/2)
    lat_max = min(90, user_lat + lat_range/2)
    lon_min = max(-180, user_lon - lon_range/2)
    lon_max = min(180, user_lon + lon_range/2)
    
    print(f"Zoom window lat/lon bounds:")
    print(f"Latitude: {lat_min} to {lat_max}")
    print(f"Longitude: {lon_min} to {lon_max}")
    
    # Convert lat/lon bounds to plot coordinates
    zoom_x_min, zoom_y_min = rotate_90_counterclockwise(lon_min, lat_min)
    zoom_x_max, zoom_y_max = rotate_90_counterclockwise(lon_max, lat_max)
    
    print(f"Zoom window plot coordinates:")
    print(f"X: {zoom_x_min} to {zoom_x_max}")
    print(f"Y: {zoom_y_min} to {zoom_y_max}")
    
    # Display background in zoom plot
    zoom_ax.imshow(background_image, origin='upper', aspect='equal', extent=[0, 1927, 0, 662])
    
    # Set zoom limits
    zoom_ax.set_xlim(zoom_x_min, zoom_x_max)
    zoom_ax.set_ylim(zoom_y_min, zoom_y_max)
    
    # Draw data on zoom plot
    draw_interpolated_smoothed_paths_with_circles(data, colors, zoom_ax)
    
    # Style zoom plot
    zoom_ax.patch.set_alpha(0.0)
    zoom_ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add border to zoom plot
    for spine in zoom_ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    # Style main plot
    main_ax.patch.set_alpha(0.0)
    main_ax.legend(loc='upper right')
    
    # Remove axes for both plots
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    zoom_ax.set_xticks([])
    zoom_ax.set_yticks([])
    
    return fig, main_ax, zoom_ax

# Example usage
colors = {
    "cw_perimeter": "#CC00CC",
    "digital_perimeter": "#00CCCC",
    "ssb_perimeter": "#CCCC00"
}

scale_x = 159  # 180%
scale_y = 301  # 118%
off_x = +470
off_y = +85

dxview_url = "https://hf.dxview.org/perspective/KM38lr"
data = fetch_zones(14, process_dxview_url(dxview_url))

# Create the plots
fig, main_ax, zoom_ax = create_plots(data, colors, dxview_url, zoom_level=3)
plt.show()