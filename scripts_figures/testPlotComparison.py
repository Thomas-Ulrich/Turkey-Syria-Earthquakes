import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import obspy
import os
import shapefile as shp
import pandas as pd
import rasterio
import argparse
import seissolxdmf

def SetupMap(ax, gridlines_left=True):
    # Setup the map with cartopy
    ax.set_extent([36, 38.8, 35.8, 38.2], crs=ccrs.PlateCarree())
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
    for fault in ["main_fault_segmented", "fault_near_hypo"]:
        fn = f"/home/ulrich/work/Mw_78_Turkey/{fault}.shp"
        sf = shp.Reader(fn)
        for sr in sf.shapeRecords():
            listx = []
            listy = []
            for xNew, yNew in sr.shape.points:
                listx.append(xNew)
                listy.append(yNew)
            ax.plot(listx, listy, "k")




def project_SeisSol_data(xdmfFilename):
    sx = seissolxdmf.seissolxdmf(xdmfFilename)
    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()
    U = sx.ReadData("u1" if args.band[0]=="EW" else "u2", sx.ndt - 1)

    # project the data to geocentric (lat, lon)
    from pyproj import Transformer

    myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
    transformer = Transformer.from_crs(myproj, "epsg:4326")
    lats, lons = transformer.transform(xyz[:, 0], xyz[:, 1])
    xy = np.vstack((lons, lats)).T

    # compute triangule barycenter
    xy_barycenter = (
        xy[connect[:, 0], :] + xy[connect[:, 1], :] + xy[connect[:, 2], :]
    ) / 3.0

    return lons, lats, connect, U


parser = argparse.ArgumentParser(description="compare displacement with geodetics")
#parser.add_argument("geodeticFilename", help="filename of the gedetic data")
parser.add_argument(
    "--downsampling",
    nargs=1,
    help="downsampling of INSar data (larger-> faster)",
    type=int,
    default=[4],
)
parser.add_argument(
    "--surface", nargs=1, help="SeisSol xdmf surface file"
)
parser.add_argument(
    "--extension", nargs=1, default=(["png"]), help="extension output file"
)
parser.add_argument(
    "--band", nargs=1, default=(["EW"]), help="EW or NS"
)
args = parser.parse_args()




fig = plt.figure()
fig.set_size_inches(10, 10)
ax = []
ax.append(fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()))
SetupMap(ax[0])

file_name = '../ThirdParty/mosaic_turkey_wgs84.tif'
with rasterio.open(file_name) as src:
    band1 = -src.read(1 if args.band[0]=="EW" else 2)
    print('Band1 has shape', band1.shape)
    height = band1.shape[0]
    width = band1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    ds = args.downsampling[0]
    xs= np.array(xs)[::ds,::ds]
    ys = np.array(ys)[::ds,::ds]
    band1 = band1[::ds,::ds]

vmax=4
vmin=-vmax
c = ax[0].pcolormesh(
    xs, ys, band1, cmap="RdBu", rasterized=True, vmin=vmin, vmax=vmax
)

if args.surface:
    ax.append(fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree()))
    SetupMap(ax[1], gridlines_left=False)
    lons, lats, connect, D_los = project_SeisSol_data(args.surface[0])
    ax[1].tripcolor(
        lons,
        lats,
        connect,
        facecolors=D_los,
        cmap="RdBu_r",
        rasterized=True,
        vmin=-vmax,
        vmax=vmax,
    )

# Add colorbar
# left, bottom, width, height
cbaxes = fig.add_axes([0.92, 0.25, 0.01, 0.25])
fig.colorbar(c, ax=ax[-1], cax=cbaxes)


plt.title(args.band[0])
plt.savefig(f"comparison_sentinel2{args.band[0]}.pdf")

plt.show()
