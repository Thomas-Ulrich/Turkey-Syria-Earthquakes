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
from geodetics_common import *

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def setup_map(ax, gridlines_left=True, draw_labels=True):
    """Setup the background map with cartopy"""
    if args.band[0] in ["azimuth", "range"]:
        extentmap = [36.6, 38.4, 36.7, 38.2]
    else:
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
            ax.plot(listx, listy, "k", linewidth=0.5)


def tree_query(arguments):
    """KD tree chunk query (find nearest node in SeisSol unstructured grid)"""
    i, q = arguments
    a = xys[chunks[i]]
    dist, index = tree.query(a)
    if q != 0:
        q.put(i)
    return index


def project_seissol_data_to_structured_grid(
    xs, ys, lonlat_barycenter, connect, displ, saveprefix=""
):
    # xs, ys: obs grid
    # x,y, connect: unstructured grid
    # U: unstructured data

    ValueNodes = np.zeros(xs.shape)

    global xys, chunks, tree
    xys = np.vstack((xs.flatten(), ys.flatten())).transpose()
    nx, ny = xs.shape
    lst = list(range(xys.shape[0]))
    nchunks = 500
    chunks = np.array_split(lst, nchunks)

    prefix, ext = os.path.splitext(os.path.basename(args.surface[0]))
    stringtime = time.ctime(os.path.getmtime(args.surface[0]))
    stringtime = "_".join(stringtime.split())
    fn = f"allindices/{saveprefix}allindices{prefix}{stringtime}{nx * ny}.npy"

    if os.path.isfile(fn):
        allindices = np.load(fn, allow_pickle=True)
    else:
        print("starting tree.query....")
        tree = spatial.KDTree(lonlat_barycenter)
        start = time.time()
        nprocs = 3
        assert nprocs <= cpu_count()
        pool = Pool(processes=nprocs)
        ma = Manager()
        q = ma.Queue()
        inputs = list(range(nchunks))
        args2 = [(i, q) for i in inputs]

        Result = pool.map_async(tree_query, args2)
        pool.close()
        while True:
            if Result.ready():
                break
            remaining = nchunks - q.qsize()
            print("Waiting for", remaining, "tasks to complete...")
            time.sleep(2.0)
        allindices = np.ascontiguousarray(np.transpose(np.array(Result.get())))
        print(np.shape(allindices))
        print("done: %f" % (time.time() - start))

        if not os.path.exists("allindices"):
            os.makedirs("allindices")
        np.save(fn, allindices)
        print("saved")

    for i in range(nchunks):
        ik = np.floor_divide(chunks[i], ny)
        jk = np.remainder(chunks[i], ny)
        ValueNodes[ik[:], jk[:]] = displ[allindices[i][:]]
    return ValueNodes


def generate_quiver_plot(lon_g, lat_g, ew, ns, ax):
    """plot the optical displacement data and synthetics with arrows"""
    nDownSample = 25
    scale = 70
    xg = lon_g[::nDownSample, ::nDownSample]
    yg = lat_g[::nDownSample, ::nDownSample]

    if args.surface:
        U1 = project_seissol_data_to_structured_grid(
            xg, yg, lonlat_barycenter, connect, U
        )
        V1 = project_seissol_data_to_structured_grid(
            xg, yg, lonlat_barycenter, connect, V
        )
        ax.quiver(
            xg,
            yg,
            U1,
            V1,
            scale=scale,
            angles="xy",
            units="width",
            color="green",
            width=0.002,
            zorder=1,
        )
    import cv2

    print("applying a median blur on data")
    ew = cv2.medianBlur(ew, 5)
    ns = cv2.medianBlur(ns, 5)
    datae = ew[::nDownSample, ::nDownSample]
    datan = ns[::nDownSample, ::nDownSample]

    ax.quiver(
        xg,
        yg,
        datae,
        datan,
        scale=scale,
        angles="xy",
        units="width",
        color="k",
        width=0.002,
        zorder=1,
    )


parser = argparse.ArgumentParser(description="compare displacement with geodetics")
parser.add_argument(
    "--downsampling",
    nargs=1,
    help="downsampling of INSar data (larger-> faster)",
    type=int,
    default=[4],
)
parser.add_argument("--surface", nargs=1, help="SeisSol xdmf surface file")
parser.add_argument(
    "--extension", nargs=1, default=(["png"]), help="extension output file"
)
parser.add_argument(
    "--band",
    nargs=1,
    default=(["EW"]),
    help="EW, NS, azimuth or range",
    choices=["EW", "NS", "azimuth", "range", "los77", "los184"],
)
parser.add_argument(
    "--noVector",
    dest="noVector",
    default=False,
    action="store_true",
    help="do not display displacement vectors",
)

parser.add_argument(
    "--diff",
    dest="diff",
    default=False,
    action="store_true",
    help="plot obs-syn instead on synthetics",
)

parser.add_argument(
    "--vmax", nargs=1, default=([4.0]), help="max of colorbar", type=float
)

args = parser.parse_args()


fig = plt.figure()
fig.set_size_inches(5, 4.5)
ax = []
ax.append(fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()))
setup_map(ax[0])

if args.band[0] in ["EW", "NS"]:
    lon_g, lat_g, ew, ns = read_optical_cc(args.downsampling[0])
    obs_to_plot = ew if args.band[0] == "EW" else ns
elif args.band[0] in ["azimuth", "range"]:
    lon_g, lat_g, obs_to_plot, phi_g, theta_g = read_scansar(
        args.band[0], args.downsampling[0]
    )

elif args.band[0] in ["los184", "los77"]:
    lon_g, lat_g, obs_to_plot, vx, vy, vz = read_insar(
        args.band[0], args.downsampling[0]
    )

vmax = args.vmax[0]
vmin = -vmax

c = ax[0].pcolormesh(
    lon_g,
    lat_g,
    obs_to_plot,
    cmap=cm.vik,
    rasterized=True,
    vmin=vmin,
    vmax=vmax,
)

if args.surface:
    lons, lats, lonlat_barycenter, connect, U, V, W = read_seissol_surface_data(
        args.surface[0]
    )
    # this is an inset axes over the main axes
    # Only work with matplotlib >= 3.6!
    ax.append(ax[0].inset_axes([0.45, 0.01, 0.54, 0.54], projection=ccrs.PlateCarree()))
    setup_map(ax[1], gridlines_left=False, draw_labels=False)

    if args.band[0] == "EW":
        syn_to_plot = U
    elif args.band[0] == "NS":
        syn_to_plot = V
    elif args.band[0] in ["azimuth", "range"]:
        syn_to_plot = compute_LOS_displacement_SeisSol_data_from_LOS_angles(
            lon_g, lat_g, theta_g, phi_g, lonlat_barycenter, args.band[0], U, V, W
        )
    if args.band[0] in ["los77", "los184"]:
        syn_to_plot = compute_LOS_displacement_SeisSol_data_from_LOS_vector(
            lon_g,
            lat_g,
            vx,
            vy,
            vz,
            lonlat_barycenter,
            U,
            V,
            W,
        )
        # syn_to_plot = np.reshape(syn_to_plot, (np.max(np.shape(vx)), np.min(np.shape(vx))))

    if args.diff:
        # interpolate satellite displacement on the unstructured grid
        obs_inter = RGIinterp(lon_g, lat_g, obs_to_plot, lonlat_barycenter)
        print(args.band[0], np.nanstd(syn_to_plot - obs_inter))

    ax[1].tripcolor(
        lons,
        lats,
        connect,
        facecolors=syn_to_plot - obs_inter if args.diff else syn_to_plot,
        cmap=cm.vik,
        rasterized=True,
        vmin=-vmax,
        vmax=vmax,
    )

if not args.noVector and args.band[0] in ["EW", "NS"]:
    generate_quiver_plot(lon_g, lat_g, ew, ns, ax[0])
# Add colorbar
# left, bottom, width, height
cbaxes = fig.add_axes([0.92, 0.25, 0.01, 0.25])
xticks = [-4, -2, 0, 2, 4] if vmax == 4.0 else [vmin, 0, vmax]
clb = fig.colorbar(c, ax=ax[-1], cax=cbaxes, ticks=xticks)
clb.ax.set_title(f"{args.band[0]} (m)", loc="left")


if not os.path.exists("output"):
    os.makedirs("output")
fn = f"output/comparison_geodetic_{args.band[0]}.{args.extension[0]}"
plt.savefig(fn, dpi=150, bbox_inches="tight")
print(f"done writing {fn}")
