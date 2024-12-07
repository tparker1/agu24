import os
import glob
import pickle
# import gc

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def load_fjord_polygons():
    """
    Load fjord polygons from a master shapefile.

    Returns:
        GeoDataFrame: Contains fjord polygons with attributes including fjordID, gatesIDs, group, latorder, glaciers.
    """
    file = os.path.expanduser('~/Data/Polygons/Greenland_Fjord_Master.zip')
    polygons = gpd.read_file(file)
    polygons['geometry'] = polygons['geometry'].convex_hull
    return polygons


def load_region_polygons():
    """
    Load extended fjord polygons from a master shapefile.

    Returns:
        GeoDataFrame: Contains extended region (multi-fjord) polygons with attributes including region id and related fjords, gates, group, and glaciers.
    """
    file = os.path.expanduser('~/Data/Polygons/Greenland_ExtRegions/Greenland_ExtRegions_Master_20240711.shp')
    polygons = gpd.read_file(file)
    return polygons


def top_fjord_gate_tuples():
    """
    Generate tuples of top fjord and gate IDs based on mean values.

    Returns:
        list of tuples: Each tuple contains a fjordID and a gateID from the top 20 entries sorted by mean values.
    """
    
    file = os.path.expanduser('~/Documents/mlml/oceancolour/analysis/overall_mean_by_fjord.csv')
    df = pd.read_csv(file)
    df = df.sort_values(by='mean', ascending=False)
    fjord_gate_tuples = list(zip(df['fjordID'].head(20), df['gateID'].head(20)))
    fjord_gate_tuples = [(fjord, gate) for fjord, gate in fjord_gate_tuples if pd.notna(fjord) and pd.notna(gate)]
    return fjord_gate_tuples

def region_names():
    return pickle.load(open(os.path.expanduser('~/Documents/mlml/oceancolour/journal/misc/region_names_map.pkl'), 'rb'))

def region_names_map():
    return {0: 'Kangerdlugssuaq',
                    1: 'Storstrommen',
                    2: 'Zachariae',
                    3: 'Qannaq',
                    4: 'Kakivfaat',
                    5: 'Upernavik',
                    6: 'Ummannaq',
                    7: 'Torsukataq'}


def group_fjord_map():
    """
    Load a dictionary mapping fjord groups from a pickle file.

    Returns:
        dict: A dictionary where keys are group names and values are corresponding data.
    """
    folder = ('~/Documents/mlml/oceancolour/stats/group_polygons.pkl')
    folder = os.path.expanduser(folder)
    with open(folder, 'rb') as f:
        groups = pickle.load(f)
    return groups

def get_group_bounds():
    """
    Calculate geographical bounds for each group based on their annual data files.

    Returns:
        dict: A dictionary with group names as keys and their geographical bounds (min_lat, max_lat, min_lon, max_lon) as values.
    """
    data_folder = os.path.expanduser('~/Data/group_annuals/')
    group_files = glob.glob(os.path.join(data_folder, 'g_*/g_*_2009.nc'))
    group_files.sort()

    group_bounds = {}
    for file in group_files:
        ds = xr.open_dataset(file)
        
        group_name = os.path.basename(file).split('_')[1]
        
        min_lat, max_lat = ds['lat'].min().item(), ds['lat'].max().item()
        min_lon, max_lon = ds['lon'].min().item(), ds['lon'].max().item()
        
        group_bounds[group_name] = {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon}
    return group_bounds

def gl_extents_4326():
    """
    Define the geographical extents of Greenland in EPSG:4326 coordinate system.

    Returns:
        list: Geographical extents [min_lon, min_lat, max_lon, max_lat].
    """
    return [-59, -29, 58, 85]

def convert_int_time(coord):
    """
    Convert integer time values to pandas datetime objects.

    Args:
        coord (array-like): Integer time values.

    Returns:
        DatetimeIndex: Converted datetime objects.
    """
    return pd.to_datetime(coord, unit='d', origin='unix')

def view_monthly_means(files):
    """
    Load and process satellite data files to view monthly means of chlorophyll-a.

    Args:
        files (list of str): Paths to satellite data files.

    Returns:
        xarray.Dataset: Dataset after processing, where chlorophyll-a values above 99 are filtered out.
    """
    g = xr.open_mfdataset(files, combine='by_coords', chunks={'time': 50})
    g = g.where(g['chlor_a'] < 99)
    g['time'] = convert_int_time(g['time'])
    return g

def custom_legends():
    """
    Print example code for creating custom legends with matplotlib.
    """
    print("""\
    blue_line = mlines.Line2D([], [], color='blue', markersize=15, label='Fjords')
    red_line = mlines.Line2D([], [], color='red', markersize=15, label='Glaciers')
          
    plt.legend(handles=[blue_line, red_line])""")
    return 

def load_gdf(file):
    """
    Load a GeoDataFrame from a file and apply convex hull to geometries.

    Args:
        file (str): Path to the file.

    Returns:
        GeoDataFrame: Loaded GeoDataFrame with modified geometries.
    """
    polygons = gpd.read_file(file)
    polygons['geometry'] = polygons['geometry'].convex_hull
    return polygons

def plot_polygons(polygons, id='id'):
    """
    Plot polygons on a stereographic projection centered on Greenland.

    Args:
        polygons (GeoDataFrame): GeoDataFrame containing polygons to plot.
        id (str): Column name to use as label for each polygon. Defaults to 'id'.
    """
    fig, ax = plt.subplots(figsize=(8,8), dpi=300, subplot_kw={'projection': ccrs.Stereographic(central_longitude=-45, central_latitude=90)})

    ax.set_extent(gl_extents_4326(), crs=ccrs.PlateCarree())

    for index, row in polygons.iterrows():
        label = row[id]

        geometry = row['geometry']
        ax.add_geometries([geometry], ccrs.PlateCarree(), label=row['id'], linewidth=0.75)

        centroid = geometry.centroid
        x_centroid, y_centroid = centroid.x, centroid.y
        
        valign, halign = 'center', 'center'
        x_text, y_text = x_centroid + 2, y_centroid + 0.05

        ax.text(x_text, y_text, label, transform=ccrs.PlateCarree(), fontsize=7, ha=halign, va=valign)

    ax.coastlines(linewidth=0.5, alpha= 0.45)

    plt.show()


def region_position():
    pos_map = {1:[3, 'a'],
           2:[2, 'f'], 
           3:[4, 'b'],
           4:[1, 'g'],
           5:[5, 'c'],
           6:[0, 'h'],
           7:[6, 'd'],
           8:[7, 'j']}
    return pos_map

