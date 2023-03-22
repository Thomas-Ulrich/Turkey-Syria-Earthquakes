import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import obspy
import os
import shapefile as shp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


fig = plt.figure()
fig.set_size_inches(3, 7)
ax = fig.add_subplot(1, 1, 1)


stations2plot1 = [
    "4611",
    "4615",
    "4625",
    "4616",
]



angle = 180 + 90 + 50
angle = 40

for fn in ["../ThirdParty/Turkey_Emergency_EQ_Data/simple_fault_2023-02-17/simple_fault_2023-2-17.shp"]:
    df = gpd.read_file(fn)
    print(df)
    glist = gpd.GeoSeries([g for g in df["geometry"]])
    glist = glist.rotate(angle, origin=(0, 0))
    glist.plot(ax=ax, color='k')

col =  ['m', 'b']
for i, stations2plot in enumerate([stations2plot1]):
    df = pd.read_csv("../ThirdParty/stations.csv")
    df = df.drop_duplicates(subset=['Code'])
    df["waveform_to_plot"] = [row["Code"] in stations2plot for index, row in df.iterrows()]
    df = df.loc[df["waveform_to_plot"] == True]

    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=4326)
    gdf = gdf.rotate(angle, origin=(0, 0))

    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, df["Code"]):
        ax.annotate(label+"  ", xy=(x, y), xytext=(3, -3), textcoords="offset points")
    ax.axis('off')

    gdf.plot(ax=ax, color=col[i])

plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.tight_layout()
#plt.show()
plt.savefig('station_rotated_sub_super.svg')
