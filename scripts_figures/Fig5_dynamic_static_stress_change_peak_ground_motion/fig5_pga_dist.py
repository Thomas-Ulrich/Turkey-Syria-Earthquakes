"""
Script for plotting PGAs and GMPE residuals vs. JB distance.

Requires:
- observed/simulated PGA CSV files
- Rupture JSON files

Writes:
- pga_dist.png
- resid_dist.png
"""

import geojson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openquake.hazardlib.imt import PGA
from impactutils.rupture import quad_rupture, origin
from openquake.hazardlib.gsim.akkar_2014 import AkkarEtAlRjb2014
from openquake.hazardlib.contexts import SitesContext, DistancesContext, RuptureContext


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
plot_only_station_available_for_both =  True

# PGA vs dist figure
fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3.5), sharey=True)

# Residuals vs dist figure
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3.5), sharey=True)


# Loop over observations and simulations
for type in ["obs", "syn"]:
    # Loop over both events
    for i, evid in enumerate(["us6000jllz", "us6000jlqa"]):
        if plot_only_station_available_for_both:
            if type == "obs":
                df = pd.read_csv(f"{type}_{evid}.csv", dtype={"codes": str})
                other_evid = "us6000jllz" if evid == "us6000jlqa" else "us6000jlqa"
                df2 = pd.read_csv(f"{type}_{other_evid}.csv", dtype={"codes": str})
                df = df.merge(df2, on="codes", suffixes=("", "obs2"))
                df.to_csv("merged_obs.csv", index=False)
            else:
                df = pd.read_csv(f"{type}_{evid}.csv", dtype={"codes": str})

            if type == "syn":
                df2 = pd.read_csv(f"merged_obs.csv", dtype={"codes": str})
                df = df.merge(df2, on="codes", suffixes=("", "obs"))
        else:
            df = pd.read_csv(f"{type}_{evid}.csv", dtype={"codes": str})
            if type == "syn":
                df2 = pd.read_csv(f"obs_{evid}.csv", dtype={"codes": str})
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
        df["dists"] = rup.computeRjb(stalats, stalons, stadeps)[0]
        df = df[df.dists < 250.0]
        dx = DistancesContext()
        dx.rjb = np.linspace(1, 1000, 1000)

        rx = RuptureContext()
        rx.mag = mags[i]
        rx.rake = 0  # assume strike slip
        sx = SitesContext()

        vs30 = 760  # assume VS30=760 m/s
        sx.sids = np.full_like(dx.rjb, 0)
        sx.vs30 = np.full_like(dx.rjb, vs30)

        imt = PGA()
        gmpe = AkkarEtAlRjb2014()
        mean = gmpe.get_mean_and_stddevs(sx, rx, dx, imt, [])[0]

        ax = axes1[i]

        # Plot GMPE
        if type == "obs":
            ax.plot(dx.rjb, 100 * np.exp(mean), c="r", label="Akkar2014 GMPE")

        # Compute GMPE residuals
        dx.rjb = np.array(df["dists"])
        sx.sids = np.full_like(dx.rjb, 0)
        sx.vs30 = np.full_like(dx.rjb, vs30)
        pred = 100 * np.exp(gmpe.get_mean_and_stddevs(sx, rx, dx, imt, [])[0])
        if type == "obs":
            obs = df["pgas"] / 9.81
            color = OBS_COLOR
            marker = OBS_MARKER
            label = "Observed data"
        else:
            obs = 100 * df["pgas"] / 9.81
            color = SIM_COLOR
            marker = SIM_MARKER
            label = "Simulated data"

        resid = np.log(obs / pred)

        # Compute binned medians
        binmeans, obsmeds = [], []
        bins = np.logspace(0, 3, base=10, num=20)
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
            ax.set_ylabel("PGA (%g)")
        ax.set_xlim(1, 1000)
        ax.set_ylim(2e-2, 2e2)
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
        ax.plot(binmeans, residmeds, c=color)
        ax.scatter(
            binmeans,
            residmeds,
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
fig1.savefig("output/pga_dist.pdf", bbox_inches="tight", dpi=300)
fig2.savefig("output/resid_dist.pdf", bbox_inches="tight", dpi=300)
