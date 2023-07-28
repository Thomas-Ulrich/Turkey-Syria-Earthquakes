import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import argparse
import matplotlib
import os


def computeMw(label, time, moment_rate):
    M0 = np.trapz(moment_rate[:], x=time[:])
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07
    print(f"{label} moment magnitude: {Mw} (M0 = {M0:.4e})")
    return Mw


ps = 8
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)
matplotlib.rcParams['lines.linewidth'] = 0.5


parser = argparse.ArgumentParser(description="plot moment rate comparison")
parser.add_argument(
    "prefix_paths", nargs="+", help="path to prefix of simulations to plots"
)
parser.add_argument("--labels", nargs="+", help="labels associated with the prefix")
parser.add_argument("--t0_2nd", nargs="+", help="origin time of 2nd event", type=float)
args = parser.parse_args()

if args.labels:
    assert len(args.prefix_paths) == len(args.labels)

if not os.path.exists("output"):
    os.makedirs("output")

fig = []
ax = []
for j, event in enumerate(["mainshock", "second_event"]):
    fig.append(plt.figure(figsize=(0.5*7.5, 0.3*7.5 * 8.0 / 16), dpi=80))
    ax.append(fig[j].add_subplot(111))

cols_mainshock = ["m", "b", "g", "y"]
cols_2nd = ["b", "m", "g", "y"]
if args.t0_2nd:
    assert(len(args.t0_2nd) == len(args.prefix_paths))
    t0_2nd = args.t0_2nd
else:
    t0_2nd = [100 for i in args.prefix_paths]
    print("t0_2nd not set, using 100s")

for i, prefix_path in enumerate(args.prefix_paths):
    df0 = pd.read_csv(f"{prefix_path}-energy.csv")
    df0 = df0.pivot_table(index="time", columns="variable", values="measurement")
    for j, event in enumerate(["mainshock", "second_event"]):
        if event == "mainshock":
            df = df0[df0.index < 85]
            print(
                "warning: Mw computed over 0-85s (avoiding contribution of small residual moment after rupture)"
            )
            cols = cols_mainshock
        else:
            #df = df0[(df0.index > 100) & (df0.index < 140)]
            df = df0[(df0.index > t0_2nd[i]) & (df0.index < t0_2nd[i]+40)]
            print(
                f"warning: Mw computed over {t0_2nd[i]}-{t0_2nd[i]+40} (avoiding contribution of small residual moment after rupture)"
            )
            if df.empty:
                print(f"no second event in {prefix_path}")
                continue
            cols = cols_2nd
        df["seismic_moment_rate"] = np.gradient(
            df["seismic_moment"], df.index[1] - df.index[0]
        )
        label = args.labels[i] if args.labels else os.path.basename(prefix_path)
        Mw = computeMw(label, df.index.values, df["seismic_moment_rate"])
        t0 = 0 if event == "mainshock" else t0_2nd[i]
        ax[j].plot(
            df.index.values - t0,
            df["seismic_moment_rate"] / 1e19,
            cols[i],
            label=f"{label} (Mw={Mw:.2f})",
        )

for j, event in enumerate(["mainshock", "second_event"]):
    Melgar = np.loadtxt(f"../../ThirdParty/moment_rate_Melgar_et_al_{event}.txt")
    ax[j].plot(
        Melgar[:, 0],
        Melgar[:, 1],
        "k",
        label="Melgar et al., 2023",
    )
    """
    Okuwaki = np.loadtxt(f"../../ThirdParty/moment_rate_Okuwaki_et_al_23_{event}.txt")
    ax[j].plot(
        Okuwaki[:, 0],
        Okuwaki[:, 1],
        "k:",
        label="Okuwaki et al., 2023",
    )
    """
    usgs = np.loadtxt(f"../../ThirdParty/moment_rate_usgs_{event}.txt")
    ax[j].plot(
        usgs[:, 0],
        usgs[:, 1]/1e19,
        "k:",
        label="USGS",
    )

    ax[j].legend(frameon=False, loc="upper right")
    if event == "mainshock":
        ax[j].set_xlim([0, 80])
    else:
        ax[j].set_xlim([0, 40])
    ax[j].set_ylim(bottom=0)

    ax[j].spines["top"].set_visible(False)
    ax[j].spines["right"].set_visible(False)
    ax[j].get_xaxis().tick_bottom()
    ax[j].get_yaxis().tick_left()

    ax[j].set_ylabel(r"moment rate (e19 $\times$ Nm/s)")
    ax[j].set_xlabel("time (s)")

    fn = f"output/moment_rate_{event}.svg"
    fig[j].savefig(fn, bbox_inches="tight", transparent=True)
    print(f"done write {fn}")
