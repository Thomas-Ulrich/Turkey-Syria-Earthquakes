import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import shapefile as shp
import rasterio
import argparse
import seissolxdmf
import time
from scipy import spatial
from multiprocessing import Pool, cpu_count, Manager
from cmcrameri import cm
import matplotlib
from pyproj import Transformer
import matplotlib.colors as colors

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def read_seissol_surface_data(xdmfFilename):
    """read unstructured free surface output and associated data.
    compute cell_barycenter"""
    sx = seissolxdmf.seissolxdmf(xdmfFilename)
    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()
    PGV = sx.ReadData("PGV", sx.ndt - 1)

    # project the data to geocentric (lat, lon)

    myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
    transformer = Transformer.from_crs(myproj, "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(xyz[:, 0], xyz[:, 1])
    xy = np.vstack((lons, lats)).T

    # compute triangule barycenter
    lonlat_barycenter = (
        xy[connect[:, 0], :] + xy[connect[:, 1], :] + xy[connect[:, 2], :]
    ) / 3.0

    return lons, lats, lonlat_barycenter, connect, PGV


def setup_map(ax, gridlines_left=True, draw_labels=True):
    """Setup the background map with cartopy"""
    extentmap = [36, 38.8, 36.0, 38.6]

    ax.set_extent(extentmap, crs=ccrs.PlateCarree())
    scale = "10m"
    ax.add_feature(
        cfeature.LAND.with_scale(scale), facecolor="whitesmoke", rasterized=True
    )
    ax.add_feature(cfeature.OCEAN.with_scale(scale), rasterized=True)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale))
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linestyle=":")
    locs = np.arange(-180, 180, 1.0)
    gl = ax.gridlines(draw_labels=draw_labels, ylocs=locs, xlocs=locs, linestyle=":")
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
            ax.plot(listx, listy, "k", linewidth=0.5)


parser = argparse.ArgumentParser(description="compare displacement with geodetics")
parser.add_argument("surface", nargs=1, help="SeisSol xdmf surface file")
parser.add_argument(
    "--extension", nargs=1, default=(["png"]), help="extension output file"
)

parser.add_argument(
    "--vmax", nargs=1, default=([4.0]), help="max of colorbar", type=float
)

args = parser.parse_args()
vmax = args.vmax[0]

fig = plt.figure()
fig.set_size_inches(5, 4.5)
ax = []
ax.append(fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()))
setup_map(ax[0])

lons, lats, lonlat_barycenter, connect, PGV = read_seissol_surface_data(args.surface[0])

c = ax[0].tripcolor(
    lons,
    lats,
    connect,
    facecolors=PGV,
    cmap=cm.vik,
    rasterized=True,
    norm=colors.LogNorm(vmin=1e-2, vmax=4),
)

# Add colorbar
# left, bottom, width, height
cbaxes = fig.add_axes([0.92, 0.25, 0.01, 0.25])
clb = fig.colorbar(c, ax=ax[-1], cax=cbaxes, norm=colors.LogNorm())
clb.ax.set_title(f"PGV (m/s)", loc="left")


if not os.path.exists("output"):
    os.makedirs("output")
fn = f"output/PGV.{args.extension[0]}"
plt.savefig(fn, dpi=150, bbox_inches="tight")
print(f"done writing {fn}")
