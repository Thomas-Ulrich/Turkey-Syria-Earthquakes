"""'
Script for reading the processed strong motion data and saving
CSV files of the PGAs and times.

Note: the "strong_motion_data" folder is available on Google drive (shared
with Thomas)
"""


import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
import groundMotionRoutines as gmr
import gme
from tqdm import tqdm

# Stations that had obvious data problems
badlist = {
    "us6000jllz": [
        "0720",
        "3113",
        "3114",
        "3117",
        "3119",
        "3120",
        "3121",
        "6102",
        "5505",
        "2713",
        "2710",
    ],
    "us6000jlqa": ["0118", "0209", "3301", "2715"],
}

org_times = {
    "us6000jllz": UTCDateTime("2023-02-06 01:17:34"),
    "us6000jlqa": UTCDateTime("2023-02-06 10:24:49"),
}

use_gmrot = True
use_PGV = False

for evid in ["us6000jllz", "us6000jlqa"]:
    t0 = org_times[evid]

    print("Reading the strong motion data for %s" % evid)
    st = read("../../ThirdParty/strong_motion_data/processed/%s/*.mseed" % evid)
    if use_PGV:
        st = st.integrate()
    for bad in badlist[evid]:
        rm = st.select(station=bad)
        for tr in rm:
            st.remove(tr)
    st.trim(starttime=t0, endtime=t0 + 250)
    st_e = st.select(channel="E")
    st_n = st.select(channel="N")

    df = pd.read_csv("../../ThirdParty/stations.csv")
    lats = []
    lons = []
    codes = []
    for tr in st_e:
        sta = tr.stats.station
        lats.append(float(df[df.codes == sta].lats))
        lons.append(float(df[df.codes == sta].lons))
        codes.append(sta)

    print("Computing the PGAs")
    times, pgas = [], []

    for tr_e, tr_n in tqdm(zip(st_e, st_n), total=len(st_e)):
        if use_gmrot:
            pgas.append(gme.compute_gmrotdpp_PGA(tr_n, tr_e))
            times.append(0.0)
        else:
            if abs(tr_n.data).max() > abs(tr_e.data).max():
                tr_l = tr_n
            else:
                tr_l = tr_e
            pgas.append(abs(tr_l.data).max())
            times.append(
                tr_l.stats.starttime + tr_l.stats.delta * np.argmax(abs(tr_l.data)) - t0
            )

    idx = np.argsort(times)
    pgas = np.array(pgas)[idx]
    times = np.array(times)[idx]
    lons = np.array(lons)[idx]
    lats = np.array(lats)[idx]
    codes = np.array(codes)[idx]
    dfn = pd.DataFrame(
        {"lats": lats, "lons": lons, "codes": codes, "pgas": pgas, "times": times}
    )
    fname = "obs_%s.csv" % evid
    dfn.to_csv(fname, index=False)
    print("Saved CSV file:", fname)
