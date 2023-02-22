import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import obspy
import os
import shapefile as shp
import pandas as pd


def SetupMap(ax, gridlines_left=True):
    # Setup the map with cartopy
    ax.set_extent([35.5, 39.5, 35.8, 38.2], crs=ccrs.PlateCarree())
    scale = "10m"
    ax.add_feature(cfeature.LAND.with_scale(scale))
    ax.add_feature(cfeature.OCEAN.with_scale(scale))
    ax.add_feature(cfeature.COASTLINE.with_scale(scale))
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linestyle=":")
    locs = np.arange(-180, 180, 2.0)
    gl = ax.gridlines(draw_labels=True, ylocs=locs, xlocs=locs)
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = gridlines_left


fig = plt.figure()
fig.set_size_inches(10, 10)
ax = []
ax.append(fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()))
SetupMap(ax[0])


stations2plot = [
    "0215",
    "0213",
    "4611",
    "4632",
    "4615",
    "4625",
    "4616",
    "2718",
    "3138",
    "3139",
    "3141",
    "3140",
]


for fault in ["main_fault_segmented", "fault_near_hypo"]:
    fn = f"../geometry/{fault}.shp"
    sf = shp.Reader(fn)
    for sr in sf.shapeRecords():
        listx = []
        listy = []
        for xNew, yNew in sr.shape.points:
            listx.append(xNew)
            listy.append(yNew)
        plt.plot(listx, listy, "k")


fn = "../ThirdParty/stations.csv"
cols = ["Code", "Longitude", "Latitude"]
df = pd.read_csv(fn)

df["waveform_to_plot"] = [row["Code"] in stations2plot for index, row in df.iterrows()]
df_no = df.loc[df["waveform_to_plot"] == False]
for index, row in df_no.iterrows():
    is_inside = (35.5 <= row["Longitude"] <= 39.5) and (35.8 <= row["Latitude"] <= 38.2)
    c = "b" if row["Code"] in stations2plot else "k"
    if row["DeviceCode"] == "N" and is_inside:
        plt.scatter(
            row["Longitude"],
            row["Latitude"],
            marker="v",
            facecolors="none",
            edgecolors=c,
        )
        # plt.text(row["Longitude"] + 0.002, row["Latitude"], row["Code"], size=8)

df_yes = df.loc[df["waveform_to_plot"]]
for index, row in df_yes.iterrows():
    is_inside = (35.5 <= row["Longitude"] <= 39.5) and (35.8 <= row["Latitude"] <= 38.2)
    if row["DeviceCode"] == "N" and is_inside:
        plt.scatter(
            row["Longitude"],
            row["Latitude"],
            marker="v",
            facecolors="none",
            edgecolors="b",
        )
        plt.text(
            row["Longitude"] + 0.04, row["Latitude"], row["Code"], size=8, color="b"
        )
plt.show()
