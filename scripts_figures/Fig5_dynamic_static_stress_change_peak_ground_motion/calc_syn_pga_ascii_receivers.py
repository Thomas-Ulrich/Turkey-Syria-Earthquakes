"""
Script compute synthetic PGAs from SeisSol ascii receivers
"""

import seissolxdmf
import numpy as np
import pandas as pd
from tqdm import trange
from pyproj import Transformer
import groundMotionRoutines as gmr
import gme
import argparse

# parsing python arguments
parser = argparse.ArgumentParser(
    description="compute synthetic PGAs from SeisSol ascii receivers"
)
parser.add_argument("folderPrefix", help="folder and output prefix")
parser.add_argument(
    "--max_component",
    dest="max_component",
    default=False,
    action="store_true",
    help="compute from the max of each component and not gmrot(geometric mean)",
)
parser.add_argument(
    "--PGV",
    dest="PGV",
    default=False,
    action="store_true",
    help="compute PGV instead of PGA",
)
args = parser.parse_args()

SPROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
TRANSFORMER = Transformer.from_crs(SPROJ, "epsg:4326", always_xy=True)
TRANSFORMER_INV = Transformer.from_crs("epsg:4326", SPROJ, always_xy=True)


folderprefix = args.folderPrefix
asciiStationFile = "../../ThirdParty/stations.csv"
sta = pd.read_csv(asciiStationFile)
stations2plot = sta["codes"].to_list()
lInvStationLookUpTable = gmr.compileInvLUTGM(
    folderprefix,
    TRANSFORMER,
    stations2plot,
    inventory=None,
    asciiStationFile=asciiStationFile,
)

use_gmrot = not args.max_component

# Time that the 1st event ends and the 2nd event begins
time_split = 150

[xsyn, ysyn, zsyn], variablelist, synth = gmr.readSeisSolSeismogram(folderprefix, 1)
dt = synth[1, 0]

split = int(time_split / dt)

for evid in ["us6000jllz", "us6000jlqa"]:
    df = pd.read_csv(asciiStationFile)
    # check is station is in refined area
    x, y = TRANSFORMER_INV.transform(df.lons, df.lats)
    ref_region_theta = np.radians(45)
    ct = np.cos(ref_region_theta)
    st = np.sin(ref_region_theta)
    ref_region_xc = 20e3
    ref_region_yc = 50e3
    df["u"] = (x - ref_region_xc) * ct + (y - ref_region_yc) * st
    df["v"] = (x - ref_region_xc) * -st + (y - ref_region_yc) * ct
    df = df[(abs(df.u) < 200e3) & (abs(df.v) < 100e3)]
    print(df)

    if evid == "us6000jllz":
        start = 0
        stop = split
    else:
        start = split
        stop = -1

    pgas = []
    times = []
    for code in df["codes"]:
        if code not in lInvStationLookUpTable:
            print(code, end=" ")
            pgas.append(np.nan)
            times.append(np.nan)
            continue
        idst = lInvStationLookUpTable[code]
        [xsyn, ysyn, zsyn], variablelist, synth = gmr.readSeisSolSeismogram(
            folderprefix, idst
        )
        u = synth[start:stop, variablelist.tolist().index("v1")]
        v = synth[start:stop, variablelist.tolist().index("v2")]

        if args.PGV:
            a1, a2 = u, v
        else:
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
    print()
    df["pgas"] = pgas
    df["times"] = times
    df = df.dropna()
    df.to_csv(f"syn_{evid}.csv", index=False)
