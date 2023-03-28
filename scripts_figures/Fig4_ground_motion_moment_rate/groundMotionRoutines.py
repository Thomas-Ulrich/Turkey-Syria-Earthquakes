import os
import numpy as np
from obspy import read, Trace
from obspy.clients.fdsn.header import FDSNException
import xml.etree.ElementTree as ET
import pyproj
import matplotlib.pyplot as plt
import glob


def ReadSeisSolSeismogram(folderprefix, idst, coords_only=False):
    """READ seissol receiver nb i"""
    mytemplate = f"{folderprefix}-receiver-{idst:05d}*"
    lFn = glob.glob(mytemplate)
    if not lFn:
        return False
    if len(lFn) > 1:
        print("warning, several files match the pattern", lFn)
    fid = open(lFn[0])
    fid.readline()
    variablelist = fid.readline()[11:].split(",")
    variablelist = np.array([a.strip().strip('"') for a in variablelist])
    xsyn = float(fid.readline().split()[2])
    ysyn = float(fid.readline().split()[2])
    zsyn = float(fid.readline().split()[2])
    if coords_only:
        return [xsyn, ysyn, zsyn]
    synth = np.loadtxt(fid)
    fid.close()
    return ([xsyn, ysyn, zsyn], variablelist, synth)


def findStationFromCoords(Client, lonlatdepth, eps=1e-3):
    """Finds stations using obspy.
    kind of Slow and may return several stations"""
    try:
        inventory = Client.get_stations(
            minlatitude=lonlatdepth[1] - eps,
            maxlatitude=lonlatdepth[1] + eps,
            minlongitude=lonlatdepth[0] - eps,
            maxlongitude=lonlatdepth[0] + eps,
        )
        code = inventory[0][0].code
    except (FDSNException, IndexError):
        return "not_a_station"
    return code


def findStationFromInventory(inventory, lonlatdepth, eps=5e-3):
    """Find code (e.g. KIKS, HSES,etc) of station from coordinates."""
    code = False
    nnet = len(inventory[:])
    for inet in range(nnet):
        for stat in inventory[inet][:]:
            lon = stat.longitude
            lat = stat.latitude
            if (abs(lon - lonlatdepth[0]) < eps) & (abs(lat - lonlatdepth[1]) < eps):
                code = stat.code
                print(code, lon, lat)
                return code
    return code


def findStationFromCoordsInFile(lxmlStationFile, lonlatdepth):
    """Find station code (e.g. KIKS, HSES,etc) of station from coordinates
    by reading station xml file"""
    for xmlf in lxmlStationFile:
        tree = ET.parse(xmlf)
        root = tree.getroot()
        for stations in root.iter("station"):
            lon = float(stations.get("lon"))
            lat = float(stations.get("lat"))
            code = stations.get("code")
            if (abs(lon - lonlatdepth[0]) < 2e-3) & (abs(lat - lonlatdepth[1]) < 2e-3):
                print(code, lon, lat)
                break
    return code


def findStationFromCoordsInRawFile(StationFile, lonlatdepth, stations2plot=[]):
    """Find station code (e.g. KIKS, HSES,etc) of station from coordinates
    by raw ascii file"""
    code = False
    ext = os.path.splitext(StationFile)[1]
    if ext == ".csv":
        import pandas as pd

        # cols = ["Code", "Longitude", "Latitude"]
        cols = ["codes", "lons", "lats"]
        stationInfo = pd.read_csv(StationFile)[cols].to_numpy()
    else:
        stationInfo = np.genfromtxt(StationFile, dtype="str", delimiter=" ")
    nstations = len(stationInfo[:, 0])
    for i in range(0, nstations):
        lat = float(stationInfo[i, 2])
        lon = float(stationInfo[i, 1])
        if (abs(lon - lonlatdepth[0]) < 1e-3) & (abs(lat - lonlatdepth[1]) < 1e-3):
            code = stationInfo[i, 0]
            # print(code, lon, lat)
            if (code in stations2plot) or (not stations2plot):
                break
    return code


def CreateObspyTraceFromSeissolSeismogram(station, variablelist, synth, starttime):
    """Load synthetics into an obspy stream"""
    st_syn = read()
    st_syn.clear()
    xyz = "ENZ"
    uvw = "uvw" if "u" in variablelist else [f"v{i}" for i in range(1, 4)]
    for i in range(0, 3):
        j = np.where(variablelist == uvw[i])[0]
        try:
            j = j[0]
        except IndexError:
            print("uvw[i] = %s not in variable list:" % (uvw[i]), variablelist)
            raise ("uvw[i] not found in variable list")
        tr = Trace()
        # tr.stats.station = station +'.'+ xyz[i]
        tr.stats.station = station
        tr.stats.channel = xyz[i]
        tr.data = synth[:, j]
        tr.stats.delta = synth[1, 0] - synth[0, 0]
        tr.stats.starttime = starttime
        st_syn.append(tr)
    return st_syn


def ImportObspyTrace(client, NetworkStation, pathObservations, starttime, endtime):
    nsarray = NetworkStation.split(".")
    if len(nsarray) == 1:
        network = "*"
        station = NetworkStation
        channel = ""
    elif len(nsarray) == 2:
        network = nsarray[0]
        station = nsarray[1]
        channel = ""
    else:
        network = nsarray[0]
        station = nsarray[1]
        channel = nsarray[2]
    channel = channel + "*"
    print(network, station)
    mytemplate = "%s/%s_%s*" % (pathObservations, station, starttime.date)
    f = os.popen("ls " + mytemplate)
    now = f.read()
    fname = now.strip()

    if os.path.isfile(fname):
        print("reading the data...")
        st_obs = read(fname)
    else:
        print("requesting the data...")
        if channel == "*":
            channel = "H*"
        # Fetch waveform from IRIS FDSN web service into a ObsPy stream object
        # and automatically attach correct response
        try:
            st_obs = client.get_waveforms(
                network=network,
                station=station,
                location="*",
                channel=channel,
                starttime=starttime,
                endtime=endtime,
                attach_response=True,
            )
        except FDSNException:
            try:
                print(station + ":No channel " + channel + " available... trying M")
                st_obs = client.get_waveforms(
                    network=network,
                    station=station,
                    location="*",
                    channel="M*",
                    starttime=starttime,
                    endtime=endtime,
                    attach_response=True,
                )
            except FDSNException:
                try:
                    print(station + ":No channel M* available... trying *")
                    st_obs = client.get_waveforms(
                        network=network,
                        station=station,
                        location="*",
                        channel="*",
                        starttime=starttime,
                        endtime=endtime,
                        attach_response=True,
                    )
                except FDSNException:
                    print(station + ":No data available for request.")
                    return False
        print(st_obs)
        # define a filter band to prevent amplifying noise during the deconvolution
        # pre_filt = (0.1, 0.25, 24.5, 25.5)
        # pre_filt = (0.005, 0.006, 30.0, 35.0)
        # st_obs.remove_response(output='VEL', pre_filt=pre_filt)
        st_obs.remove_response(output="VEL")

        has_East = False
        for tr in st_obs:
            if tr.stats.component == "E":
                has_East = True

        if not has_East:
            inv = client.get_stations(
                network=network,
                station=station,
                starttime=starttime,
                endtime=endtime,
                level="response",
            )
            st_obs.rotate(method="->ZNE", inventory=inv)

        fname = (
            f"{pathObservations}/{station}_{starttime.format_iris_web_service()}.mseed"
        )
        st_obs.write(fname, format="MSEED")
    return st_obs


def PlotComparisonStationXYZ(
    station, lTrace, lColors, fplotname, starttime, dispLabel=False
):
    # Compare for a station observation and simulation on the same figure (3 components)
    fig0, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    xyz = "xyz"
    components = ["E", "N", "Z"]
    for j, ax in enumerate([ax1, ax2, ax3]):
        comp = components[j]
        for it, myTrace in enumerate(lTrace):
            if np.amax(myTrace.select(component=comp)[0].data) == 0.0:
                continue
            my_time = myTrace.select(component=comp)[0].times(reftime=starttime)
            y = myTrace.select(component=comp)[0].data
            ax.plot(
                my_time,
                y,
                lColors[it],
            )
            ax.set_xlim(0.0, my_time[-1])
        if dispLabel:
            ax.set_ylabel("D%s (m)" % xyz[j])
        else:
            ax.set_ylabel("V%s (m/s)" % xyz[j])

    ax1.set_title(station)
    ax3.set_xlabel("time")
    fig0.set_size_inches(18.5, 10.5)

    plt.savefig(fplotname, bbox_inches="tight")
    # plt.show()
    plt.close()


def ComputeCosineTapper(t):
    from math import cos, pi

    cosinetapperS = np.ones(t.shape)
    # we tapper the data to have cleaner spectrograms
    a = 2.0
    ids = np.where(t < a)
    for ki in ids[0]:
        cosinetapperS[ki] = 0.5 * (1 - cos(pi * t[ki] / a))
    tf = t[-1]
    ids = np.where(t > (tf - a))
    for ki in ids[0]:
        cosinetapperS[ki] = 0.5 * (1 - cos(pi * (tf - t[ki]) / a))
    return cosinetapperS


def PlotSpectrograms(
    station,
    lTrace,
    lColors,
    fplotprefix,
    idStation,
    lonlatdepth,
    plot_trace_below=False,
    components=["E", "N", "Z"],
):
    # Compare for a station observation and simulation on the same figure (3 components)
    ncomponents = len(components)
    nrows = 2 if plot_trace_below else 1
    wlen = 20.0
    for it, myTrace in enumerate(lTrace):
        fig0, lax = plt.subplots(
            nrows,
            ncomponents,
            sharex=True,
            sharey=False,
            squeeze=0,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for j, comp in enumerate(components):
            myTrace.select(component=comp)[0].spectrogram(
                log=True, title=comp, show=False, axes=lax[0, j], wlen=wlen
            )

            lax[0, j].set_title(
                f"receiver {idStation} ({lonlatdepth[0]:.3f}, {lonlatdepth[1]:.3f}): {comp}"
            )
            if plot_trace_below:
                lax[1, j].plot(
                    myTrace.select(component=comp)[0].times(),
                    myTrace.select(component=comp)[0].data,
                    "k",
                )
        lax[-1, 0].set_xlabel("time (s)")
        lax[0, 0].set_ylabel("frequency (Hz)")
        lax[0, 0].set_ylim(bottom=1.0 / wlen, top=1.0)

        # fig0.set_size_inches(18.5, 10.5)
        fig0.set_size_inches(ncomponents * 7, 6)
        fname = f"{fplotprefix}_{wlen}s_{it}.png"
        plt.savefig(fname, bbox_inches="tight")
        print(f"done generating {fname}")
        plt.close()


def PlotSpectraComparisonStationXYZ(station, lTrace, lColors, fplotname):
    # Compare for a station observation and simulation on the same figure (3 components)
    fig0, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    xyz = "xyz"
    for j, ax in enumerate([ax1, ax2, ax3]):
        # Compare signal on the same time segment
        tmin, tmax = -1e10, 1e10
        for it, myTrace in enumerate(lTrace):
            t = myTrace[j].times()
            tmin = max(tmin, np.amin(t))
            tmax = min(tmax, np.amax(t))
        t0 = 0.5 * (tmax - tmin)
        t1 = 0.5 * (tmax + tmin)

        for it, myTrace in enumerate(lTrace):
            t = myTrace[j].times()
            ids = np.where(abs(t - t1) < t0)[0]
            t = t[ids]
            cosinetapperS = ComputeCosineTapper(t)
            sp = np.fft.fft(myTrace[j].data[ids] * cosinetapperS)
            freq = np.fft.fftfreq(n=t.shape[-1], d=t[1] - t[0])
            nf = np.shape(freq)[0] // 2
            ax.loglog(freq[0:nf], np.abs(sp)[0:nf], lColors[it])

    for j, ax in enumerate([ax1, ax2, ax3]):
        ax.set_ylim(bottom=1e-5)
        ax.set_xlabel("freq")
        ax.set_ylabel("V%s (m/s)" % xyz[j])

    ax1.set_title(station)
    # fig0.set_size_inches(18.5, 10.5)
    fig0.set_size_inches(20, 6)
    plt.savefig(fplotname, bbox_inches="tight")
    plt.close()


def removeTopRightAxis(ax):
    # remove top right axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def compileInvLUTGM(
    folderprefix,
    idlist,
    transformer,
    stations2plot=[],
    inventory=None,
    RawStationFile=None,
):
    print("compiling lookup table...")
    StationLookUpTable = {}
    id_not_found = []
    for i, idStation in enumerate(idlist):
        # Load SeisSol and obspy traces
        xyzs = ReadSeisSolSeismogram(folderprefix, idStation, coords_only=True)
        if not xyzs:
            id_not_found.append(idStation)
            continue
        lonlatdepth = transformer.transform(xyzs[0], xyzs[1], xyzs[2])
        station = False
        if inventory:
            station = findStationFromInventory(inventory, lonlatdepth)
        if not station:
            station = findStationFromCoordsInRawFile(
                RawStationFile, lonlatdepth, stations2plot
            )
        if station:
            StationLookUpTable[idStation] = station
    if id_not_found:
        print(f"no SeisSol receiver with id {id_not_found}")
    print(StationLookUpTable)
    invStationLookUpTable = {v: k for k, v in StationLookUpTable.items()}
    return invStationLookUpTable
