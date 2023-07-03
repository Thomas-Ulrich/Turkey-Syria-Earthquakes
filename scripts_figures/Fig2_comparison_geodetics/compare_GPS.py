import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapefile as shp
import pandas as pd
import argparse
import seissolxdmf
from pyproj import Transformer
import matplotlib
import scipy.interpolate as interp
from pathlib import Path
import os

# from matplotlib import font_manager
# font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)

# Custom font for station name
fpath = Path("/usr/local/share/fonts/Poppins-Light.ttf")
if not os.path.isfile(fpath):
    fpath = "sans"


def setup_map(ax, extentmap, gridlines_left=True, draw_labels=True):
    """Setup the background map with cartopy"""
    ax.set_extent(extentmap, crs=ccrs.PlateCarree())
    scale = "10m"
    ax.add_feature(
        cfeature.LAND.with_scale(scale), facecolor="whitesmoke", rasterized=True
    )
    ax.add_feature(cfeature.OCEAN.with_scale(scale), rasterized=True)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale))
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linestyle=":")
    locs = np.arange(-180, 180, 1.0)
    gl = ax.gridlines(draw_labels=draw_labels, ylocs=locs, xlocs=locs)
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = gridlines_left
    for fn in [
        "../../ThirdParty/Turkey_Emergency_EQ_Data/simple_fault_2023-02-17/simple_fault_2023-2-17.shp"
    ]:
        sf = shp.Reader(fn)
        for sr in sf.shapeRecords():
            listx = []
            listy = []
            for xNew, yNew in sr.shape.points:
                listx.append(xNew)
                listy.append(yNew)
            ax.plot(listx, listy, "k", linewidth=0.7)


def read_seissol_surface_data(xdmfFilename, event):
    """read unstructured free surface output and associated data.
    compute cell_barycenter"""
    sx = seissolxdmf.seissolxdmf(xdmfFilename)
    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()

    if event == 1:
        U = sx.ReadData("u1", 0)
        V = sx.ReadData("u2", 0)
        W = sx.ReadData("u3", 0)
    elif event == 2:
        U = sx.ReadData("u1", 1) - sx.ReadData("u1", 0)
        V = sx.ReadData("u2", 1) - sx.ReadData("u2", 0)
        W = sx.ReadData("u3", 1) - sx.ReadData("u3", 0)

    # project the data to geocentric (lat, lon)
    myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
    transformer = Transformer.from_crs(myproj, "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(xyz[:, 0], xyz[:, 1])
    xy = np.vstack((lons, lats)).T

    # compute triangule barycenter
    lonlat_barycenter = (
        xy[connect[:, 0], :] + xy[connect[:, 1], :] + xy[connect[:, 2], :]
    ) / 3.0

    return lons, lats, lonlat_barycenter, connect, U, V, W


def interpolate_seissol_surf_output(lonlat_barycenter, U, df):
    """interpolate SeisSol free surface output to GPS data location"""

    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, U)
    locGPS = np.vstack((df["lon"].to_numpy(), df["lat"].to_numpy())).T

    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, U)
    ui = Fvsm.__call__(locGPS)
    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, V)
    vi = Fvsm.__call__(locGPS)
    Fvsm = interp.LinearNDInterpolator(lonlat_barycenter, W)
    wi = Fvsm.__call__(locGPS)

    return ui, vi, wi


def plot_quiver(lon, lat, ew, ns, scale, color, width, label):
    ax[0].quiver(
        lon,
        lat,
        ew,
        ns,
        scale=scale,
        angles="xy",
        units="width",
        color=color,
        width=width,
        zorder=1,
        label=label,
    )


parser = argparse.ArgumentParser(
    description="compare simulated surface displacement with GPS"
)

parser.add_argument("--surface", nargs=1, help="SeisSol xdmf surface file")

parser.add_argument(
    "--event",
    nargs=1,
    default=(["1"]),
    help="Plot gps comparison for the Mw 7.8 (1) or for the Mw 7.7 events (2)",
    choices=["1", "2"],
)

parser.add_argument(
    "--component",
    nargs=1,
    default=(["horizontal"]),
    help="horizontal or vertical",
    choices=["horizontal", "vertical"],
)

arg = parser.parse_args()

print(f"Plot {arg.component[0]} GPS comparison for event {arg.event[0]}")

event = np.int64(arg.event[0])

# Read GPS data
gps = "../../ThirdParty/gps_turkey_sequence.csv"
df = pd.read_csv(gps)

# Read SeisSol output and interpolate output to GPS data point locations
lons, lats, lonlat_barycenter, connect, U, V, W = read_seissol_surface_data(
    arg.surface[0], event
)
ui, vi, wi = interpolate_seissol_surf_output(lonlat_barycenter, U, df)

if event == 1:
    # data
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    ew = df["ew1"].to_numpy()
    ns = df["ns1"].to_numpy()
    dz = df["dz1"].to_numpy()

    # plotting parameters for event 1
    if arg.component[0] == "horizontal":
        scale = 2
        leg = 0.3
    else:
        scale = 1
        leg = 0.2
    width = 0.006

elif event == 2:
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    ew = df["ew2"].to_numpy()
    ns = df["ns2"].to_numpy()
    dz = df["dz2"].to_numpy()

    # plotting parameters for event 2
    if arg.component[0] == "horizontal":
        scale = 10
        leg = 2
    else:
        scale = 6
        leg = 1
    width = 0.006

# Plot GPS comparison
extentmap = [35.4, 39, 35.8, 39]
color1 = [1, 0.7, 0.3]  # color for data
color2 = [0.2, 0.5, 1.0]  # color for predictions

fig = plt.figure()
fig.set_size_inches(5, 4.5)
ax = []
ax.append(fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()))
setup_map(ax[0], extentmap)

# Plot data
if arg.component[0] == "horizontal":
    plot_quiver(lon, lat, ew, ns, scale, color1, width, "data")
    plot_quiver(lon, lat, ui, vi, scale, color2, width, "prediction")

elif arg.component[0] == "vertical":
    plot_quiver(lon, lat, wi * 0, wi, scale, color2, width, "prediction")
    plot_quiver(lon, lat, dz * 0, dz, scale, color1, width, "data")


# Plot station name
for i in range(0, np.size(df, 0) - 1):
    if (
        (df["lon"].to_numpy()[i] > extentmap[0])
        and (df["lon"].to_numpy()[i] < extentmap[1])
        and (df["lat"].to_numpy()[i] > extentmap[2])
        and (df["lat"].to_numpy()[i] < extentmap[3])
    ):
        if df["station"][i] == "AKLE":
            ax[0].text(
                df["lon"].to_numpy()[i] - 0.28,
                df["lat"].to_numpy()[i] + 0.07,
                df["station"][i],
                font=fpath,
                color=[0.2, 0.2, 0.2],
            )
        else:
            ax[0].text(
                df["lon"].to_numpy()[i] + 0.07,
                df["lat"].to_numpy()[i],
                df["station"][i],
                font=fpath,
                color=[0.2, 0.2, 0.2],
            )

# Plot legend
plot_quiver(38, 36, leg, 0, scale, color2, width, "data")
ax[0].text(
    38 - 0.1,
    36,
    "Predictions",
    font=fpath,
    color=color2,
    horizontalalignment="right",
    verticalalignment="center",
)
plot_quiver(38, 36.3, leg, 0, scale, color1, width, "data")
ax[0].text(
    38 - 0.1,
    36.3,
    "Data",
    font=fpath,
    color=color1,
    horizontalalignment="right",
    verticalalignment="center",
)
ax[0].text(
    38 + 0.4,
    36.15,
    f"{leg} m",
    font=fpath,
    color=[0.3, 0.3, 0.3],
    horizontalalignment="right",
    verticalalignment="center",
)

# Compute RMS
def nanrms(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))

rms1 = nanrms(ui - ew)
rms2 = nanrms(vi - ns)
rms3 = nanrms(wi - dz)

rmstot = np.sqrt(rms1**2 + rms2**2 + rms3**2)
print('RMS GPS:',np.round(rmstot,5),'m')

# Write .png file
fn = f"output/comp_GPS_event_{event}_comp_{arg.component[0]}.png"
plt.savefig(fn, dpi=300, bbox_inches="tight")
print(f"done writing {fn}")
