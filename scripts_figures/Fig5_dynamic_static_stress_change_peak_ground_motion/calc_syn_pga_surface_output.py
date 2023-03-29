"""
Script compute synthetic PGAs from SeisSol free surface output.
"""

import seissolxdmf
import numpy as np
import pandas as pd
from tqdm import trange
from pyproj import Transformer
import argparse

SPROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
TRANSFORMER = Transformer.from_crs(SPROJ, "epsg:4326", always_xy=True)
TRANSFORMER_INV = Transformer.from_crs("epsg:4326", SPROJ, always_xy=True)

# Only calculate synthetic PGAs for stations within this bounding box
lonmin = 34
lonmax = 40
latmin = 35
latmax = 39

parser = argparse.ArgumentParser(
    description="compute synthetic PGAs from SeisSol ascii receivers"
)
parser.add_argument("folderPrefix", help="folder and output prefix")
parser.add_argument(
    "--PGA_from_max_component",
    dest="PGA_from_max_component",
    default=False,
    action="store_true",
    help="compute PGA from the max of each component and not gmrot(geometric mean)",
)
args = parser.parse_args()

SURFACE_OUTPUT_PATH = args.folderPrefix + "-surface.xdmf"

use_gmrot = not args.PGA_from_max_component

sx = seissolxdmf.seissolxdmf(SURFACE_OUTPUT_PATH)
geo = sx.ReadGeometry()
connect = sx.ReadConnect()
coords = geo[connect].mean(axis=1)
dt = sx.ReadTimeStep()

# Time that the 1st event ends and the 2nd event begins
time_split = 150
split = int(time_split / dt)

for evid in ["us6000jllz", "us6000jlqa"]:
    df = pd.read_csv("../../ThirdParty/stations.csv")
    df = df[
        (df.lons > lonmin)
        & (df.lons < lonmax)
        & (df.lats > latmin)
        & (df.lats < latmax)
    ]

    x, y = TRANSFORMER_INV.transform(df.lons, df.lats)
    sta_coords = np.vstack((x, y)).T
    idxs = []
    for sta in sta_coords:
        # Find the nearest point for this receiver
        dists = np.linalg.norm(sta - coords[:, :-1], axis=1)
        idxs.append(dists.argmin())

    if evid == "us6000jllz":
        start = 0
        stop = split
    else:
        start = split
        stop = -1

    pgas = []
    times = []
    for i in trange(len(idxs)):
        idx = idxs[i]
        u = sx.ReadDataChunk("v1", firstElement=idx, nchunk=1, idt=-1)[start:stop, 0]
        v = sx.ReadDataChunk("v2", firstElement=idx, nchunk=1, idt=-1)[start:stop, 0]

        # Differentiate to get acceleration
        a1 = np.gradient(u, dt)
        a2 = np.gradient(v, dt)

        if use_gmrot:
            pgas.append(gme.compute_gmrotdpp_PGA(a1, a2))
            times.append(0.0)
        else:
            # Use greater of two horizontals for PGA
            if abs(a1).max() > abs(a2).max():
                pgas.append(abs(a1).max())
                times.append(abs(a1).argmax() * dt)
            else:
                pgas.append(abs(a2).max())
                times.append(abs(a2).argmax() * dt)

    df["pgas"] = pgas
    df["times"] = times
    df.to_csv("syn_%s.csv" % evid, index=False)
