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
import time
from scipy import spatial
from multiprocessing import Pool, cpu_count, Manager
from pyproj import Transformer
from cmcrameri import cm
from scipy.interpolate import RegularGridInterpolator
import matplotlib

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def setup_map(ax, gridlines_left=True, draw_labels=True):
    """Setup the background map with cartopy"""
    ax.set_extent([36, 38.8, 36.0, 38.2], crs=ccrs.PlateCarree())
    scale = "10m"
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor='whitesmoke', rasterized=True)
    ax.add_feature(cfeature.OCEAN.with_scale(scale), rasterized=True)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale))
    ax.add_feature(cfeature.BORDERS.with_scale(scale), linestyle=":")
    locs = np.arange(-180, 180, 1.0)
    gl = ax.gridlines(draw_labels=draw_labels, ylocs=locs, xlocs=locs)
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = gridlines_left
    for fn in [
        "../ThirdParty/Turkey_Emergency_EQ_Data/simple_fault_2023-02-17/simple_fault_2023-2-17.shp"
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


def read_observation_data_one_band(fn):
    with rasterio.open(fn) as src:
        ew = src.read(1)
        print("band 1 has shape", ew.shape)
        ds = args.downsampling[0]
        height, width = ew.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        lon_g, lat_g = rasterio.transform.xy(src.transform, rows, cols)
        lon_g = np.array(lon_g)[::ds, ::ds]
        lat_g = np.array(lat_g)[::ds, ::ds]
        ew = ew[::ds, ::ds]
        return lon_g, lat_g, ew


def read_optical_cc_data(fn):
    with rasterio.open(fn) as src:
        ew = src.read(1)
        ns = src.read(2)
        print("band 1 has shape", ew.shape)
        height, width = ew.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        lon_g, lat_g = rasterio.transform.xy(src.transform, rows, cols)
        ds = args.downsampling[0]
        lon_g = np.array(lon_g)[::ds, ::ds]
        lat_g = np.array(lat_g)[::ds, ::ds]
        ew = ew[::ds, ::ds]
        ns = ns[::ds, ::ds]
    return lon_g, lat_g, ew, ns


def compute_LOS_displacement_SeisSol_data(
    lon_g, lat_g, theta_g, phi_g, lonlat_barycenter, band
):
    # interpolate satellite angles on the unstructured grid
    f = RegularGridInterpolator(
        (lon_g[0, :], lat_g[:, 0]), theta_g.T, bounds_error=False, fill_value=np.nan
    )
    theta_inter = f(lonlat_barycenter)
    g = RegularGridInterpolator(
        (lon_g[0, :], lat_g[:, 0]), phi_g.T, bounds_error=False, fill_value=np.nan
    )
    phi_inter = g(lonlat_barycenter)
    # compute displacement line of sight
    # phi azimuth, theta: range
    if band == "azimuth":
        D_los = U * np.sin(phi_inter) + V * np.cos(phi_inter)
    else:
        D_los = W * np.cos(theta_inter) + np.sin(theta_inter) * (
            U * -np.cos(phi_inter) + V * np.sin(phi_inter)
        )
        # D_los = W * np.sin(theta_inter) + np.cos(theta_inter) * (U * np.cos(phi_inter) + V * np.sin(phi_inter))
    return -D_los


def read_seissol_surface_data(xdmfFilename):
    """read unstructured free surface output and associated data.
    compute cell_barycenter"""
    sx = seissolxdmf.seissolxdmf(xdmfFilename)
    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()
    U = sx.ReadData("u1", sx.ndt - 1)
    V = sx.ReadData("u2", sx.ndt - 1)
    W = sx.ReadData("u3", sx.ndt - 1)

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
    """
    import pandas as pd
    df = pd.read_csv('../ThirdParty/coseismic_offsets.txt')
    ax.quiver(
        df['Lon'].to_numpy(),
        df['Lat'].to_numpy(),
        df['de(m)'].to_numpy(),
        df['dn(m)'].to_numpy(),
        scale=scale,
        angles="xy",
        units="width",
        color="r",
        width=0.002,
        zorder=1,
    )
    """


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
    choices=["EW", "NS", "azimuth", "range"],
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
    # Mathilde newest cc results
    fn = "../ThirdParty/Turquie_detrended_EW_NLM_destripe_wgs84.tif"
    lon_g, lat_g, ew = read_observation_data_one_band(fn)
    fn = "../ThirdParty/Turquie_detrended_NS_NLM_destripe_wgs84.tif"
    lon_g, lat_g, ns = read_observation_data_one_band(fn)
    obs_to_plot = ew if args.band[0] == "EW" else ns
elif args.band[0] in ["azimuth", "range"]:
    # Mathilde initial cc results
    fn = f"../ThirdParty/Displacement_TUR_20230114_20230207_1529_Data/20230114_HH_20230207_HH.spo_{args.band[0]}.filtered.geo.tif"
    lon_g, lat_g, obsLOS = read_observation_data_one_band(fn)
    obs_to_plot = obsLOS
    fn = "../ThirdParty/Displacement_TUR_20230114_20230207_1529_Data/20230114_HH_lv_phi.geo.tif"
    lon_g, lat_g, phi_g = read_observation_data_one_band(fn)
    fn = "../ThirdParty/Displacement_TUR_20230114_20230207_1529_Data/20230114_HH_lv_theta.geo.tif"
    lon_g, lat_g, theta_g = read_observation_data_one_band(fn)

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
    ax.append(ax[0].inset_axes([0.45, 0.01, 0.54, 0.54], projection=ccrs.PlateCarree()))
    setup_map(ax[1], gridlines_left=False, draw_labels=False)

    if args.band[0] == "EW":
        syn_to_plot = U
    elif args.band[0] == "NS":
        syn_to_plot = V
    elif args.band[0] in ["azimuth", "range"]:
        syn_to_plot = compute_LOS_displacement_SeisSol_data(
            lon_g, lat_g, theta_g, phi_g, lonlat_barycenter, args.band[0]
        )

    if args.diff:
        # interpolate satellite displacement on the unstructured grid

        f = RegularGridInterpolator(
            (lon_g[0, :], lat_g[:, 0]),
            obs_to_plot.T,
            bounds_error=False,
            fill_value=np.nan,
        )

    ax[1].tripcolor(
        lons,
        lats,
        connect,
        facecolors=syn_to_plot - f(lonlat_barycenter) if args.diff else syn_to_plot,
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
fig.colorbar(c, ax=ax[-1], cax=cbaxes)

plt.title(args.band[0])
fn = f"comparison_geodetic_{args.band[0]}.{args.extension[0]}"
plt.savefig(fn, dpi=100, bbox_inches="tight")
print(f"done writing {fn}")
# plt.show()
