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
    "--event",
    nargs=1,
    help="1: mainshock, 2: aftershock, both: both",
    choices=["both", "1", "2"],
    default=["both"],
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
PGA_PGV = "PGV" if args.PGV else "PGA"

# Time that the 1st event ends and the 2nd event begins
time_split = 150.0

[xsyn, ysyn, zsyn], variablelist, synth = gmr.readSeisSolSeismogram(folderprefix, 1)
dt = synth[1, 0]

split = int(time_split / dt)

if args.event[0] == "both":
    print(f"computing for both events, with time split {time_split}")
    levid = ["us6000jllz", "us6000jlqa"]
elif args.event[0] == "1":
    print(f"computing for mainshock only (whole simulation)")
    levid = ["us6000jllz"]
else:
    print(f"computing for 2nd event only (whole simulation)")
    levid = ["us6000jlqa"]

for evid in levid:
    df = pd.read_csv(asciiStationFile)
    # check is station is in refined area
    x, y = TRANSFORMER_INV.transform(df.lons, df.lats)

    ref_region_theta = np.radians(0.)
    ct = np.cos(ref_region_theta)
    st = np.sin(ref_region_theta)
    mesh = '175M'
    if mesh == '175M' or args.PGV:
        ref_region_xc = 50e3
        ref_region_yc = 60e3
        hu = 380e3
        hv = 380e3
    elif mesh == '31M':
        ref_region_xc = 20e3
        ref_region_yc = 50e3
        hu = 200e3
        hv = 100e3
    else:
        raise NotImplementedError
    df["u"] = (x - ref_region_xc) * ct + (y - ref_region_yc) * st
    df["v"] = (x - ref_region_xc) * -st + (y - ref_region_yc) * ct
    print(f'selecting only stations in refined area ({mesh} mesh)')
    df = df[(abs(df.u) < hu) & (abs(df.v) < hv)]
    print(df)

    if args.event == "both":
        if evid == "us6000jllz":
            start = 0
            stop = split
        else:
            start = split
            stop = -1
    else:
        start = 0
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
    df.to_csv(f"syn_{evid}_{PGA_PGV}.csv", index=False)
