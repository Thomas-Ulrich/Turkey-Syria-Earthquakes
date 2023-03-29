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
args = parser.parse_args()

SPROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
TRANSFORMER = Transformer.from_crs(SPROJ, "epsg:4326", always_xy=True)
TRANSFORMER_INV = Transformer.from_crs("epsg:4326", SPROJ, always_xy=True)


folderprefix = args.folderPrefix
RawStationFile = "../../ThirdParty/stations.csv"
sta = pd.read_csv(RawStationFile)
stations2plot = sta["codes"].to_list()
idlist = range(1, 509)
lInvStationLookUpTable = gmr.compileInvLUTGM(
    folderprefix,
    idlist,
    TRANSFORMER,
    stations2plot,
    inventory=None,
    RawStationFile=RawStationFile,
)

# Only calculate synthetic PGAs for stations within this bounding box
lonmin = 34
lonmax = 40
latmin = 35
latmax = 39

use_gmrot = True

# Time that the 1st event ends and the 2nd event begins
time_split = 150

[xsyn, ysyn, zsyn], variablelist, synth = gmr.ReadSeisSolSeismogram(folderprefix, 1)
dt = synth[1, 0]

split = int(time_split / dt)

for evid in ["us6000jllz", "us6000jlqa"]:
    df = pd.read_csv(RawStationFile)
    df = df[
        (df.lons > lonmin)
        & (df.lons < lonmax)
        & (df.lats > latmin)
        & (df.lats < latmax)
    ]
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
        [xsyn, ysyn, zsyn], variablelist, synth = gmr.ReadSeisSolSeismogram(
            folderprefix, idst
        )
        u = synth[start:stop, variablelist.tolist().index("v1")]
        v = synth[start:stop, variablelist.tolist().index("v2")]
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
    df = df.dropna()
    df.to_csv(f"syn_{evid}.csv", index=False)
