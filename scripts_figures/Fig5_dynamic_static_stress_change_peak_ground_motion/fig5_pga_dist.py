"""
Script for plotting PGAs and GMPE residuals vs. JB distance.

Requires:
- observed/simulated PGA CSV files
- Rupture JSON files

Writes:
- pga_dist.png
- pga_resid_dist.png
"""

import geojson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openquake.hazardlib.imt import PGA, PGV
from impactutils.rupture import quad_rupture, origin
from openquake.hazardlib.gsim.akkar_2014 import AkkarEtAlRjb2014
from openquake.hazardlib.contexts import SitesContext, DistancesContext, RuptureContext
from openquake.hazardlib.const import StdDev
import argparse

# Matplotlib settings
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.1
plt.rcParams["savefig.dpi"] = 500
plt.rcParams["savefig.bbox"] = "tight"

OBS_COLOR = "k"
OBS_MARKER = "o"

SIM_COLOR = "dodgerblue"
SIM_MARKER = "s"

evids = ["us6000jllz", "us6000jlqa"]
evlats = [37.230, 38.008]
evlons = [37.019, 37.211]
mags = [7.8, 7.7]

# parsing python arguments
parser = argparse.ArgumentParser(description="plot fig 5")
parser.add_argument(
    "--only_stations_both_events",
    dest="only_stations_both_events",
    default=False,
    action="store_true",
    help="plot only data of the stations that recorded both events",
)
parser.add_argument(
    "--obs_if_synthetics_available",
    dest="obs_if_synthetics_available",
    default=False,
    action="store_true",
    help="plot only obs data of the stations which are included in the synthetics dataset",
)
parser.add_argument(
    "--extension", nargs=1, default=(["png"]), help="extension output file"
)

parser.add_argument(
    "--PGV",
    dest="PGV",
    default=False,
    action="store_true",
    help="plot PGV instead of PGA",
)
args = parser.parse_args()


plot_only_station_available_for_both = args.only_stations_both_events
use_PGV = args.PGV
max_distance = 1e3

# PGA vs dist figure
fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3.5), sharey=True)

# Residuals vs dist figure
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3.5), sharey=True)
PGA_PGV = "PGV" if use_PGV else "PGA"


# Loop over observations and simulations
for type in ["obs", "syn"]:
    # Loop over both events
    for i, evid in enumerate(["us6000jllz", "us6000jlqa"]):
        df = pd.read_csv(f"{type}_{evid}_{PGA_PGV}.csv", dtype={"codes": str})
        if type == "obs" and plot_only_station_available_for_both:
            # in this case we need to generate a merged database of station that recorded both
            other_evid = "us6000jllz" if evid == "us6000jlqa" else "us6000jlqa"
            df2 = pd.read_csv(
                f"{type}_{other_evid}_{PGA_PGV}.csv", dtype={"codes": str}
            )
            df = df.merge(df2, on="codes", suffixes=("", "obs2"))
            df.to_csv("merged_obs.csv", index=False)

        if type == "obs" and args.obs_if_synthetics_available:
            df2 = pd.read_csv(f"syn_{evid}_{PGA_PGV}.csv", dtype={"codes": str})
            df = df.merge(df2, on="codes", suffixes=("", "syn"))

        if type == "syn":
            fn = "merged_obs.csv" if plot_only_station_available_for_both else f"obs_{evid}_{PGA_PGV}.csv"
            df2 = pd.read_csv(fn, dtype={"codes": str})
            df = df.merge(df2, on="codes", suffixes=("", "obs"))

        stalats = np.array(df["lats"])
        stalons = np.array(df["lons"])
        stadeps = np.full_like(stalats, 0.0)

        # Create shakemap origin
        ev = {
            "id": evids[i],
            "netid": "",
            "network": "",
            "lat": evlats[i],
            "lon": evlons[i],
            "depth": "",
            "locstring": "",
            "mag": "",
            "time": "",
            "mech": "",
            "reference": "",
            "productcode": "",
        }
        org = origin.Origin(ev)

        # Create rupture object
        with open("rupture_%s.json" % evids[i]) as f:
            gj = geojson.load(f)
        rup = quad_rupture.QuadRupture(gj, org)

        # Compute JB distances
        df["dists"] = rup.computeRjb(lon=stalons, lat=stalats, depth=stadeps)[0]
        df = df[df.dists < max_distance]
        print('using minimum distance of 1.2')
        df['dists'] = df['dists'].apply(lambda x: max(1.2, x))
        if i==0:
            clipped = ['0208', '0210', '0214', '0215', '2707', '2709',
                       '3144', '4413', '4629', '4630', '4631', '7901']
        else:
            clipped = ['2304', '4001', '4209', '4408', '4631']
        df = df[~df['codes'].isin(clipped)]
        dx = DistancesContext()
        dx.rjb = np.linspace(1, 1000, 1000)

        rx = RuptureContext()
        rx.mag = mags[i]
        rx.rake = 0  # assume strike slip
        sx = SitesContext()

        vs30 = 760  # assume VS30=760 m/s
        sx.sids = np.full_like(dx.rjb, 0)
        sx.vs30 = np.full_like(dx.rjb, vs30)

        imt = PGV() if use_PGV else PGA()
        gmpe = AkkarEtAlRjb2014()
        mean, sd = gmpe.get_mean_and_stddevs(sx, rx, dx, imt, [StdDev.TOTAL])

        ax = axes1[i]
        cm2m = 100.0
        factor = 1.0 / cm2m if use_PGV else cm2m
        # Plot GMPE
        if type == "obs":
            ax.plot(dx.rjb, factor * np.exp(mean), c="r", label="Akkar2014 GMPE")
            lower = factor * np.exp(mean - 2 * sd[0])
            upper = factor * np.exp(mean + 2 * sd[0])
            ax.fill_between(dx.rjb, lower, upper, color='r', alpha=0.05)

        # Compute GMPE residuals
        dx.rjb = np.array(df["dists"])
        sx.sids = np.full_like(dx.rjb, 0)
        sx.vs30 = np.full_like(dx.rjb, vs30)
        pred = factor * np.exp(gmpe.get_mean_and_stddevs(sx, rx, dx, imt, [])[0])
        g = 9.81
        if type == "obs":
            obs = df["pgas"] / cm2m if use_PGV else df["pgas"] / g
            color = OBS_COLOR
            marker = OBS_MARKER
            label = "observed"
        else:
            obs = df["pgas"] if use_PGV else cm2m * df["pgas"] / g
            color = SIM_COLOR
            marker = SIM_MARKER
            label = "simulated"

        resid = np.log(obs / pred)

        # Compute binned medians
        binmeans, obsmeds = [], []
        bins = np.logspace(0, 3, base=10, num=20)
        bins = bins[bins>5.0]
        for j in range(len(bins) - 1):
            binmin = bins[j]
            binmax = bins[j + 1]
            binmean = (binmin + binmax) / 2
            idx = (df.dists > binmin) & (df.dists < binmax)
            inbin = obs[idx]
            if len(inbin) > 3:
                obsmeds.append(inbin.median())
                binmeans.append(binmean)

        # Plot binned medians
        ax.scatter(
            binmeans,
            obsmeds,
            edgecolors="k",
            facecolors=color,
            s=40,
            linewidths=0.5,
            marker=marker,
            zorder=10,
            label=label,
        )
        """
        # label each point
        if type == "obs":
            for dis, pg, code in zip(df.dists, obs, df.codes):
                ax.text(max(1.5,dis), max(2e-3, pg), code, c=color, fontsize=5)
        """
        # Plot data
        ax.scatter(
            df.dists,
            obs,
            s=3,
            marker=marker,
            edgecolors=color,
            facecolors="None",
            linewidths=0.5,
        )
        ax.plot(binmeans, obsmeds, c=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("R$_{\mathrm{JB}}$ (km)")
        if i == 0:
            if use_PGV:
                ax.set_ylabel("PGV (m/s)")
            else:
                ax.set_ylabel("PGA (%g)")
        ax.set_xlim(1, 1000)
        if use_PGV:
            ax.set_ylim(2e-3, 1e1)
        else:
            ax.set_ylim(2e-2, 5e2)

        ax.set_xlim(1, 1e3)

        ax.tick_params(axis="both", which="major", labelsize=11)

        if i == 0:
            letter = "A"
        else:
            letter = "B"
        ax.text(0, 1.03, letter, transform=ax.transAxes)

        # Plot residuals
        ax = axes2[i]
        ax.scatter(
            df.dists,
            resid,
            edgecolors=color,
            facecolors="none",
            s=3,
            marker=marker,
            linewidths=0.5,
        )
        ax.set_xscale("log")
        ax.set_xlabel("R$_{\mathrm{JB}}$ (km)")
        ax.set_ylim(-4.5, 4.5)
        ax.axhline(0, ls="--", lw=0.5, c="k")

        if type == "obs":
            ax.text(
                0.02,
                0.98,
                "observed mean$=%.2f$" % resid.mean(),
                ha="left",
                va="top",
                transform=ax.transAxes,
                color=color,
            )
        else:
            ax.text(
                0.02,
                0.9,
                "simulated mean$=%.2f$" % resid.mean(),
                ha="left",
                va="top",
                transform=ax.transAxes,
                color=color,
            )
        if i == 0:
            ax.set_ylabel("ln(obs/pred)")
        ax.set_xlim(1, 1e3)

        # Compute binned residuals
        binmeans, residmeds, residstds = [], [], []
        bins = np.logspace(0, 3, base=10, num=20)

        for j in range(len(bins) - 1):
            binmin = bins[j]
            binmax = bins[j + 1]
            binmean = (binmin + binmax) / 2
            idx = (df.dists > binmin) & (df.dists < binmax)
            inbin = resid[idx]
            if len(inbin) > 3:
                residmeds.append(inbin.mean())
                residstds.append(inbin.std())
                binmeans.append(binmean)
        #we manually remove the first bin
        ax.plot(binmeans[1:], residmeds[1:], c=color)

        ax.scatter(
            binmeans[1:],
            residmeds[1:],
            edgecolors="k",
            facecolors=color,
            s=40,
            linewidths=0.5,
            marker=marker,
            zorder=10,
        )

        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.text(0, 1.03, letter, transform=ax.transAxes)

handles, labels = axes1[0].get_legend_handles_labels()
axes1[0].legend(handles, labels, loc="lower left", fontsize=9)
fig1.tight_layout()
fig2.tight_layout()
prefix = "pgv" if use_PGV else "pga"
sadd =  "_obs_if_synthetics_available" if args.obs_if_synthetics_available else ""
sadd += "_only_stations_both_events" if args.only_stations_both_events else ""
ext = args.extension[0]
fn = f"output/{prefix}_dist{sadd}.{ext}"
fig1.savefig(fn, bbox_inches="tight", dpi=300)
print(f"done writing {fn}")
fn = f"output/{prefix}_resid_dist_{sadd}.{ext}"
fig2.savefig(fn, bbox_inches="tight", dpi=300)
print(f"done writing {fn}")
