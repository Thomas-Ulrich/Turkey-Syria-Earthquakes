"""
Script compute synthetic PGAs from SeisSol free surface output.
"""

import seissolxdmf
import numpy as np
import pandas as pd
from tqdm import trange
from pyproj import Transformer
import groundMotionRoutines as gmr
import gme

SPROJ = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
TRANSFORMER = Transformer.from_crs(SPROJ, "epsg:4326", always_xy=True)
TRANSFORMER_INV = Transformer.from_crs("epsg:4326", SPROJ, always_xy=True)


folderprefix = "/home/ulrich/trash/receiversTS23/Turkey78"
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
        & (df.lons < latmax)
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

        pgas.append(gme.compute_gmrotdpp_PGA(a1, a2))
        times.append(0.0)

    df["pgas"] = pgas
    df["times"] = times
    df = df.dropna()
    df.to_csv(f"syn_{evid}.csv", index=False)