import obspy
from obspy import UTCDateTime
from obspy.core.inventory import Inventory, Network, Station, Site
from obspy.clients.fdsn import Client, RoutingClient
import os
from pyproj import Transformer
import matplotlib.pyplot as plt
import groundMotionRoutines as gmr
import numpy as np
from obspy import read
import matplotlib

ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def InitializeSeveralStationsFigure(
    nstations, ncol_per_component=1, directions=["EW", "NS", "UD"]
):
    ncomp = len(directions)
    nrow = int(np.ceil(nstations / ncol_per_component))
    ncol = ncol_per_component * ncomp
    figall, axarr = plt.subplots(
        nrow,
        ncol,
        figsize=(4.0 * ncol, nrow * 1.4),
        dpi=160,
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    Surface_waves_signa_plot = True
    axarr[-1, -1].set_xlabel("time (s)")
    for j in range(ncol):
        axarr[0, j].set_title(directions[j // ncol_per_component])
        for i in range(nrow):
            axi = axarr[i, j]
            gmr.removeTopRightAxis(axi)
            axi.tick_params(axis="x", zorder=3)
            if i < nrow - 1 and Surface_waves_signa_plot:
                axi.spines["bottom"].set_visible(False)
                axi.set_xticks([])
            if j * nrow + i >= ncomp * nstations:
                axi.spines["left"].set_visible(False)
                axi.spines["bottom"].set_visible(False)
                axi.set_xticks([])
                axi.set_yticks([])
            axi.set_xlim((0, 100))
    return [figall, axarr]


components = ["E", "N", "Z"]
# components = ["E", "N"]
ncol_per_component = 1

t1 = UTCDateTime(2023, 2, 6, 1, 17, 35.0)
t2 = t1 + 250
tplot_max = 100.0

pathObservations = "../../ThirdParty/strong_motion_data/processed/us6000jllz"
use_filter = True
# use_filter=False

RawStationFile = "../../ThirdParty/stations.csv"

stations2plot = [
    "4404",
    "0213",
    "4611",
    "4615",
    "4625",
    "4616",
    "2712",
    "2718",
    "3138",
    "3139",
    "3141",
    "3136",
]

comp2dir = {"E": "EW", "N": "NS", "Z": "UD"}
directions = [comp2dir[comp] for comp in components]
figall, axarr = InitializeSeveralStationsFigure(
    len(stations2plot), ncol_per_component, directions
)


lFolderprefix = [
    "/home/ulrich/trash/receiversTS23b/Turkey_2events_31mio_o5_2603_s205",
]

use_filter = True
plot_spectras = False
plot_spectrograms = False

plotdir = "./output/"
ext = "svg"


# setting up projections
lla = "epsg:4326"
myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37 +lat_0=37"

os.makedirs(plotdir, exist_ok=True)


# add Ground motion stations to inventory if one station is GM

print("compiling lookup table...")
lInvStationLookUpTable = []
inventory = None
nsyn = len(lFolderprefix)
transformer = Transformer.from_crs(myproj, lla, always_xy=True)
for folderprefix in lFolderprefix:
    lInvStationLookUpTable.append(
        gmr.compileInvLUTGM(folderprefix, transformer, asciiStationFile=RawStationFile)
    )

figall, axarr = InitializeSeveralStationsFigure(
    len(stations2plot), ncol_per_component, directions
)

nrows = axarr.shape[0]


def compute_j0(i, j, nrows, ncol_per_component):
    return j * ncol_per_component + i // nrows


for i, station in enumerate(stations2plot):
    i0 = i % nrows
    fn = f"{pathObservations}/20230206011732_{station}_ap_Acc.mseed"
    if not os.path.isfile(fn):
        fn = f"{pathObservations}/20230206011732_{station}_mp_Acc.mseed"
    st_obs = read(fn, format="MSEED")
    # scale to m/s
    for j, comp in enumerate(["E", "N", "Z"]):
        st_obs.select(component=comp)[0].data *= 0.01

    st_obs.integrate()

    aSt_syn = []
    for idsyn in range(nsyn):
        try:
            idStation = lInvStationLookUpTable[idsyn][station]
            xyzs, variablelist, synth = gmr.readSeisSolSeismogram(
                lFolderprefix[idsyn], idStation
            )
            lonlatdepth = transformer.transform(xyzs[0], xyzs[1], xyzs[2])
            tplot_max = np.amax(synth[0, :])
            print(station, xyzs)
            syn_str = gmr.createObspyTraceFromSeissolReceiverData(
                station, variablelist, synth, t1
            )
            aSt_syn.append(syn_str)
        except KeyError:
            print("station %s not found in SeisSol receivers" % station)
            st_zero = st_obs.copy()
            for tr in st_zero:
                tr.data *= 0.0
            aSt_syn.append(st_zero)

    if not st_obs:
        continue

    lColors = ["k", "m", "b", "g", "c"]

    lTrace = [st_obs]
    for idsyn in range(nsyn):
        lTrace.append(aSt_syn[idsyn])
    if plot_spectras:
        fplotname = "./{plotdir}/%s_%s.png" % ("spectra", station)
        gmr.PlotSpectraComparisonStationXYZ(station, lTrace, lColors, fplotname)
    if plot_spectrograms:
        fplotprefix = f"./{plotdir}/spectrogram_{station}"
        gmr.PlotSpectrograms(
            station,
            lTrace,
            lColors,
            fplotprefix,
            idStation,
            lonlatdepth,
            plot_trace_below=True,
            components=["N"],
        )

    # plot stations XYZ temporal comparisons
    if use_filter:
        for myst in lTrace:
            # myst.filter('lowpass', freq=1.0, corners=2, zerophase=True)
            myst.filter(
                "bandpass", freqmin=0.005, freqmax=1.0, corners=2, zerophase=True
            )
        filterdesc = "_bp_5e-3-5e-1"
        # filterdesc = 'lp1.0Hz'
    else:
        filterdesc = "NoFilter"
    fplotname = f"{plotdir}/%s_%s.{ext}" % (station, filterdesc)
    gmr.PlotComparisonStationXYZ(station, lTrace, lColors, fplotname, t1)

    # plot all stations on same plot
    shift = 0
    for j, comp in enumerate(components):
        shift = max(
            shift,
            1.2
            * max(
                np.amax(lTrace[0].select(component=comp)[0].data),
                -np.amin(lTrace[1].select(component=comp)[0].data),
            ),
        )
    # shift=0
    for j, comp in enumerate(components):
        j0 = compute_j0(i, j, nrows, ncol_per_component)
        for it, myTrace in enumerate(lTrace):
            if np.amax(myTrace.select(component=comp)[0].data) == 0.0:
                continue
            axarr[i0, j0].plot(
                myTrace.select(component=comp)[0].times(reftime=t1),
                myTrace.select(component=comp)[0].data + it * shift,
                lColors[it],
            )
            if ncol_per_component > 1 or j0 == 0:
                axarr[i0, j0].set_ylabel(station)

for j in range(axarr.shape[1]):
    figall.align_ylabels(axarr[:, j])
figall.subplots_adjust(wspace=0.3 if ncol_per_component > 1 else 0.2)

figall.savefig(f"{plotdir}/allDxyz{filterdesc}_mainshock.{ext}", bbox_inches="tight")
