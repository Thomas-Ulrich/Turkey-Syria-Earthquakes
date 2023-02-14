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


ps = 20
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


parser = argparse.ArgumentParser(description="plot Jason comparison")
parser.add_argument(
    "prefix_paths", nargs="+", help="path to prefix of simulations to plots"
)
parser.add_argument("--labels", nargs="+", help="labels associated with the prefix")
args = parser.parse_args()

if args.labels:
    assert len(args.prefix_paths) == len(args.labels)

fig = plt.figure(figsize=(16, 6), dpi=80)
ax = fig.add_subplot(111)
cols = ["k", "r", "g", "b", "y"]

for i, prefix_path in enumerate(args.prefix_paths):
    df = pd.read_csv(f"{prefix_path}-energy.csv")
    df = df.pivot_table(index="time", columns="variable", values="measurement")
    df["seismic_moment_rate"] = np.gradient(df["seismic_moment"], df.index[1])
    label = args.labels[i] if args.labels else os.path.basename(prefix_path)
    plt.plot(df.index.values, df["seismic_moment_rate"] / 1e19, cols[i], label=label)
    computeMw(label, df.index.values, df["seismic_moment_rate"])

plt.legend(frameon=False)
plt.xlim([0, 100])
plt.ylim(bottom=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.set_ylabel(r"moment rate (e19 $\times$ Nm/s)")
ax.set_xlabel("time (s)")

fn = "plots/moment_rate.png"
plt.savefig("plots/moment_rate.png", bbox_inches="tight")
print(f"done write {fn}")
