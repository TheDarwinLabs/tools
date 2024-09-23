import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import requests
import io
import time
import os
import zipfile
import json
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pyproj
from matplotlib.colors import LinearSegmentedColormap
import cmocean
from matplotlib.font_manager import FontProperties
import csv

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Load country mappings and languages
# country_mapping = pd.read_csv('country_mapping.csv', index_col='code')['country'].to_dict()

# Load population data
population_data = pd.read_csv('country_populations.csv', index_col='code')

# Load country languages
country_languages = pd.read_csv('country_languages.csv', index_col='code')['language'].to_dict()

def get_language(country_code):
    return country_languages.get(country_code, 'Unknown')

# Function to get or download the shapefile
@timeit
def get_shapefile(url, local_filename):
    local_dir = os.path.dirname(local_filename)
    shapefile_name = os.path.splitext(os.path.basename(local_filename))[0]
    
    # Check if the shapefile already exists
    if os.path.exists(local_filename):
        print(f"Reading shapefile from local cache: {local_filename}")
        return gpd.read_file(local_filename)
    else:
        # Check if the zip file exists
        zip_filename = os.path.join(local_dir, f"{shapefile_name}.zip")
        if os.path.exists(zip_filename):
            print(f"Extracting shapefile from local zip: {zip_filename}")
            with zipfile.ZipFile(zip_filename) as zip_ref:
                zip_ref.extractall(local_dir)
        else:
            print(f"Downloading shapefile from: {url}")
            response = requests.get(url)
            with open(zip_filename, 'wb') as f:
                f.write(response.content)
            print(f"Shapefile downloaded to: {zip_filename}")
            with zipfile.ZipFile(zip_filename) as zip_ref:
                zip_ref.extractall(local_dir)
        
        print(f"Shapefile extracted to: {local_filename}")
        return gpd.read_file(local_filename)

# URL and local filename for the shapefile
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
local_filename = "ne_110m_admin_0_countries.shp"

# Download and load world shapefile
world = get_shapefile(url, local_filename)

# Function to sample points from different countries
@timeit
def sample_points(num_points, max_attempts=100):
    points = []
    country_counts = {}
    total_population = sum(
        population_data.loc[row['ADM0_A3'], 'population']
        for _, row in world.iterrows()
        if row['ADM0_A3'] in population_data.index
    )
    
    # Project the world geometries to Equal Earth projection
    equal_earth = pyproj.CRS.from_epsg(8857)  # EPSG code for Equal Earth projection
    world_projected = world.to_crs(equal_earth)
    
    # Calculate total land area using projected geometries
    total_land_area = world_projected['geometry'].area.sum()
    
    # Load country economy data
    economy_data = pd.read_csv('country_economy.csv', index_col='code')
    
    # Calculate desired number of points for each country based on population
    country_point_allocation = {}
    per_country_points = {}
    for country, row in world_projected.iterrows():
        country_code = row['ADM0_A3']
        country_population = population_data.loc[country_code, 'population'] if country_code in population_data.index else 0
        country_area = row['geometry'].area
        desired_points = max(0, int((country_population / total_population) * num_points))
        
                # Ensure large countries have at least (country_area/total_land_area * 0.5) points
        if country_population > 10000000:
            min_points_by_area = int((country_area / total_land_area) * 0.5 * num_points)
            desired_points = max(desired_points, min_points_by_area)
        # Apply tier-based multiplier
        if country_code in economy_data.index:
            tier = economy_data.loc[country_code, 'tier']
            if tier == 'tier1':
                desired_points = int(desired_points * 1.3)
            elif tier == 'tier3':
                desired_points = int(desired_points * 0.8)
        
        # Ensure at least 20 points for tier 1 countries with population > 10M
        if tier == 'tier1' and country_population > 10000000:
            desired_points = max(desired_points, 20)
    
        
        # Adjust points for specific countries (keeping your existing adjustments)
        if country_code == 'CHN':
            desired_points = int(desired_points / 1.5)
        elif country_code == 'IND':
            desired_points = int(desired_points / 3)
        elif country_code == "BGD":
            desired_points = int(desired_points / 5)
        elif country_code == "CAN":
            desired_points = max(desired_points, 30)
        elif country_code == "RUS":
            desired_points = int(desired_points * 0.75)
        
        desired_points = min(300, desired_points)
        country_point_allocation[country_code] = desired_points

    # Rest of the function remains the same
    for country, row in world.iterrows():
        country_code = row['ADM0_A3']
        desired_points = country_point_allocation[country_code]
        
        # Get bounding box of the country
        minx, miny, maxx, maxy = row.geometry.bounds
        
        # Restrict latitude to -70 to +70 range
        miny = max(miny, -70)
        maxy = min(maxy, 70)
        
        country_points = 0
        attempts = 0
        cp = []
        while country_points < desired_points and attempts < max_attempts * desired_points:
            # Generate a random point within the bounding box
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            
            # Check if the point is within the country's geometry and within -70 to +70 latitude
            if row.geometry.contains(point) and -70 <= y <= 70:
                points.append((x, y, country_code))
                cp.append(({"LATITUDE": y, "LONGITUDE": x, "idx" : len(points)-1}))
                country_points += 1
            
            attempts += 1
        
        if country_points < desired_points:
            print(f"Warning: Only sampled {country_points}/{desired_points} points for {country_code}")
        
        country_counts[country_code] = country_points
        per_country_points[country_code] = cp

    return per_country_points, points, country_counts

# Update the total_points variable
total_points = 2000

# Call the updated sample_points function
# sampled_points, country_counts = sample_points(total_points)

@timeit
def create_graph(points):
    G = nx.Graph()
    for i, (x, y, country_code) in enumerate(points):
        language = get_language(country_code)
        G.add_node(i, pos=(x, y), language=language)
    return G

@timeit
def connect_nodes_optimized(G):
    positions = np.array([G.nodes[node]['pos'] for node in G.nodes()])
    languages = np.array([G.nodes[node]['language'] for node in G.nodes()])
    tree = cKDTree(positions)
    
    edges = []
    for i, (pos, lang) in enumerate(zip(positions, languages)):
        # Query the KD-tree for points within 400km
        indices = tree.query_ball_point(pos, r=5)
        
        # Filter points with the same language and sort by distance
        same_lang_indices = [j for j in indices if j != i and languages[j] == lang]
        distances = [np.linalg.norm(positions[j] - pos) for j in same_lang_indices]
        sorted_indices = [j for _, j in sorted(zip(distances, same_lang_indices))]
        
        # Connect to at most 6 nearest points
        for j in sorted_indices[:6]:
            edges.append((list(G.nodes())[i], list(G.nodes())[j]))
    
    G.add_edges_from(edges)
    return G

# Update the color palette with a deck-appropriate color scheme
deck_palette = [
    "#4285F4",  # Google Blue
    "#EA4335",  # Google Red
    "#FBBC05",  # Google Yellow
    "#34A853",  # Google Green
    "#FF6D00",  # Orange
    "#46BDC6",  # Teal
    "#7E57C2",  # Purple
    "#EC407A",  # Pink
    "#5D6D7E",  # Slate
    "#16A085",  # Emerald
    "#D35400",  # Pumpkin
    "#27AE60",  # Nephritis
    "#2980B9",  # Belize Hole
    "#8E44AD",  # Wisteria
    "#2C3E50",  # Midnight Blue
]

@timeit
def draw_map_optimized(G, color_dict):
    for hemisphere in ['west', 'east']:
        # Increase figure size
        fig = plt.figure(figsize=(20, 20), facecolor='none')
        
        # Use Orthographic projection with adjusted central longitude
        proj = ccrs.Orthographic(central_longitude=-90 if hemisphere == 'west' else 90, central_latitude=0)
        ax = fig.add_subplot(1, 1, 1, projection=proj)

        # Set the global extent to show the entire hemisphere
        ax.set_global()

        # Set the axes background to transparent
        ax.set_facecolor('none')

        # Prepare node data
        node_positions = np.array([G.nodes[node]['pos'] for node in G.nodes()])
        node_colors = [color_dict[G.nodes[node]['language']] for node in G.nodes()]

        # Draw edges with matching node colors, thinner lines
        for (n1, n2) in G.edges():
            pos1 = G.nodes[n1]['pos']
            pos2 = G.nodes[n2]['pos']
            color = color_dict[G.nodes[n1]['language']]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color=color, 
                    linewidth=0.5, alpha=1, transform=ccrs.Geodetic(), zorder=1)

        # Draw nodes with larger dots (increased size from 20 to 40)
        ax.scatter(node_positions[:, 0], node_positions[:, 1], s=30, c=node_colors, alpha=0.7, 
                   transform=ccrs.PlateCarree(), edgecolors='none', zorder=2)

        # Comment out legend creation
        """
        legend_elements = []
        for lang in unique_languages:
            if lang in color_dict:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=lang,
                                       markerfacecolor=color_dict[lang], markersize=10))
            else:
                print(f"Warning: No color assigned for language '{lang}'")

        # Adjust legend position and size
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
                  title="Languages", title_fontsize=18, fontsize=16, frameon=True, 
                  facecolor='none', edgecolor='#D3D3D3')
        """

        # Comment out title
        """
        plt.title(f'Global SoulLink Network - {hemisphere.capitalize()} Hemisphere', color='black', fontsize=40, 
                  fontweight='bold', fontfamily='serif', pad=30)
        """

        plt.tight_layout()
        plt.savefig(f'global_language_network_{hemisphere}_hemisphere_dark.png', dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
        plt.close()

# Main execution
start_time = time.time()

# Load data and create graph
world = get_shapefile(url, local_filename)
per_country_points, sampled_points, country_counts = sample_points(total_points)
G = create_graph(sampled_points)
G = connect_nodes_optimized(G)

print(G.edges())

# # import the csv file with country_code,language
# language_file = 'country_languages.csv'
# with open(language_file) as f:
#     reader = csv.reader(f)
#     # skip first row
#     next(reader)
#     country_languages = {row[0]: row[1] for row in reader}


major_languages = ['English', 'Chinese', 'Spanish', 'Arabic', 'Hindi', 'Bengali', 'Portuguese', 'Russian', 'Japanese', 'German', 'French', 'Italian', 'Korean', 'Turkish']

# print(country_languages)
color_dict = {lang: deck_palette[i] for i, lang in enumerate(major_languages)}
print(color_dict)

polygon_file = 'ne_110m_admin_0_countries.geojson'

with open(polygon_file) as f:
    data = json.load(f)
    # create a dictionary to store the country code and the official language
    edges = G.edges()
    for country in data['features']:
        country_code = country['properties']['ADM0_A3']
        language = country_languages.get(country_code, 'Unknown')
        points = per_country_points.get(country_code, [])
        points_idx = {point['idx']: point for point in points}
        arcs = []
        for id0, id1 in edges:
            if id0 in points_idx and id1 in points_idx:
                arcs.append({"start_lat": sampled_points[id0][1], "start_lon": sampled_points[id0][0], "end_lat": sampled_points[id1][1], "end_lon": sampled_points[id1][0]})
            
        if language in major_languages:
            color = color_dict[language]
        else:
            color = '#B0BEC5'

        country['properties']['pointColor'] = color
        country['properties']['language'] = language
        country['properties']['points'] = points
        country['properties']['arcs'] = arcs



# save the updated geojson file
output_file = 'ne_110m_admin_0_countries_language.geojson'
with open(output_file, 'w') as f:
    json.dump(data, f)
    print(f'Saved the updated geojson file to {output_file}')