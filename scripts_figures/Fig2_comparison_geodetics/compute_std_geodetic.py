import numpy as np
import os
import rasterio
import argparse
import seissolxdmf
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from geodetics_common import *

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
    "--band",
    nargs=1,
    default=(["EW"]),
    help="EW, NS, azimuth or range",
    choices=["EW", "NS", "azimuth", "range", "los77", "los184"],
)

args = parser.parse_args()

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

if args.surface:
    lons, lats, lonlat_barycenter, connect, U, V, W = read_seissol_surface_data(
        args.surface[0]
    )
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

    # interpolate satellite displacement on the unstructured grid
    obs_inter = RGIinterp(lon_g, lat_g, obs_to_plot, lonlat_barycenter)
    print(args.band[0], nanrms(syn_to_plot - obs_inter))
