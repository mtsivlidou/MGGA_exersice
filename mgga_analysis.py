import pandas as pd
import numpy as np
#%matplotlib inline # command ensures that plots will display inline in Jupyter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# path to files
file_path_gps  = r"C:\Users\d85335mt\OneDrive - The University of Manchester\Desktop\MGGA_exersice\20250317-113726 - Test.txt"
file_path_mgga = r"C:\Users\d85335mt\OneDrive - The University of Manchester\Desktop\MGGA_exersice\micro2002-01-01_f0044.txt"

# read files as pandas dataframes
gps_data  = pd.read_csv(file_path_gps)
mgga_data = pd.read_csv(file_path_mgga)

# Clean up column names (strip leading/trailing spaces)
mgga_data.columns = mgga_data.columns.str.strip()
# Strip spaces from string values in all columns
# This ensures no leading/trailing spaces remain in string-based columns
mgga_data = mgga_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

print (mgga_data.columns)
print (gps_data.columns)

print (gps_data.shape)
print (mgga_data.shape)

# convert dates to pandas datetime objects 
gps_data['date time'] = pd.to_datetime(gps_data['date time'], format='%Y-%m-%d %H:%M:%S')
# mgga_data['Time'] = pd.to_datetime(mgga_data['Time'], format='%Y-%m-%d %H:%M:%S')

###############################################################
# here I copy the timestamp to make the script run 
# it needs corrections to either interpolate to this timestamp 
# or if they match do nothing just combine the data 
n_gps = gps_data.shape[0]
mgga_data = mgga_data.iloc[0:n_gps,:]
print (mgga_data.shape)
mgga_data['Time'] = gps_data['date time']
################################################################

# add latitude and longitude info to measurements 
mgga_data['latitude'] = gps_data['latitude']
mgga_data['longitude'] = gps_data['longitude']
print (mgga_data.head(10))
print (gps_data.head(10))

# plot histogram for CH4 
plt.figure(figsize=(8, 6))
plt.hist(mgga_data['[CH4]_ppm'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of CH4')
plt.xlabel('CH4 (ppm)')
plt.ylabel('Frequency')
plt.show()

# Plot Time Series for CH4 
plt.figure(figsize=(8, 6))
plt.plot(mgga_data['Time'], mgga_data['[CH4]_ppm'], color='green', marker='o', linestyle='-')
plt.title('CH4 Time Series')
plt.xlabel('Time')
plt.ylabel('CH4 (ppm)')
plt.show()

# Plot Histogram for CO2
plt.figure(figsize=(8, 6))
plt.hist(mgga_data['[CO2]_ppm'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of CO2')
plt.xlabel('CO2 (ppm)')
plt.ylabel('Frequency')
plt.show()

# Plot Time Series for CO2 
plt.figure(figsize=(8, 6))
plt.plot(mgga_data['Time'], mgga_data['[CO2]_ppm'], color='green', marker='o', linestyle='-')
plt.title('CO2 Time Series')
plt.xlabel('Time')
plt.ylabel('CO2 (ppm)')
plt.show()

import plotly.graph_objects as go

# Create a Plotly figure with Mapbox tiles
fig = go.Figure(go.Scattermapbox(
    lat= mgga_data['latitude'],  # Use the latitude values from your DataFrame
    lon= mgga_data['longitude'],  # Use the longitude values from your DataFrame
    mode='markers',
    marker={'size': 12, 'color': mgga_data['[CH4]_ppm'], 'colorscale': 'Viridis', 'colorbar': {'title': 'CH4 Levels'}},  # Color by CH4 values
    text= mgga_data['[CH4]_ppm'],  # Display CH4 value on hover
))
# Set up Mapbox style and layout
fig.update_layout(
    mapbox_style="open-street-map",  # Use OpenStreetMap tiles for the base map
    mapbox_zoom=12,  # Zoom level
    mapbox_center={"lat": mgga_data['latitude'].mean(), "lon": mgga_data['longitude'].mean()},  # Center map on the average lat/lon
    title="City Locations with CH4 Levels",
)
# Show the map
fig.show()

# Create a Plotly figure with Mapbox tiles
fig = go.Figure(go.Scattermapbox(
    lat= mgga_data['latitude'],  # Use the latitude values from your DataFrame
    lon= mgga_data['longitude'],  # Use the longitude values from your DataFrame
    mode='markers',
    marker={'size': 12, 'color': mgga_data['[CO2]_ppm'], 'colorscale': 'Viridis', 'colorbar': {'title': 'CO2 Levels'}},  # Color by CH4 values
    text= mgga_data['[CO2]_ppm'],  # Display CH4 value on hover
))
# Set up Mapbox style and layout
fig.update_layout(
    mapbox_style="open-street-map",  # Use OpenStreetMap tiles for the base map
    mapbox_zoom=12,  # Zoom level
    mapbox_center={"lat": mgga_data['latitude'].mean(), "lon": mgga_data['longitude'].mean()},  # Center map on the average lat/lon
    title="City Locations with CO2 Levels",
)
# Show the map
fig.show()