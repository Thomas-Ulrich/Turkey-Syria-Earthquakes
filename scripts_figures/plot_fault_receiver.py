import numpy as np
import matplotlib.pylab as plt
import glob
import matplotlib

ps = 8
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def read_fault_receiver(fname):
    fid = open(fname)
    fid.readline()
    variables = [a.strip().strip('"') for a in fid.readline().split("=")[-1].split(",")]
    print(variables)
    fr = dict()
    fr["x"] = float(fid.readline().split()[-1])
    fr["y"] = float(fid.readline().split()[-1])
    fr["z"] = float(fid.readline().split()[-1])

    data = np.loadtxt(fid)
    fid.close()
    for i, name in enumerate(variables):
        fr[name] = data[:, i]
    return fr

size = 2.5
fig = plt.figure(figsize=(size, size * 8 / 16), dpi=80)
ax = fig.add_subplot(111)

for folderprefix in ["/home/ulrich/trash/test1_180_subshear_fr"]: #, "/home/ulrich/trash/test1_180_subshear_fr2"]:
    #folderprefix = "/home/ulrich/trash/test1_180_subshear_fr"
    idst = 13
    if folderprefix == "/home/ulrich/trash/test1_180_subshear_fr2":
        idst = 15
    #folderprefix = "/home/ulrich/trash/test1_180_subshear_fr2"
    #idst = 13
    if folderprefix == "/home/ulrich/trash/test1_180_subshear_fr2":
        ls = '--'
    else:
        ls = '-'


    mytemplate = f"{folderprefix}-faultreceiver-{idst:05d}*"
    lFn = glob.glob(mytemplate)
    print(lFn)
    fname = lFn[0]
    fr = read_fault_receiver(fname)

    cohesion = 1 + 0.1 * (fr["z"] + 6000.0) / 6000.0
    ax.plot(
        fr["Time"],
        np.sqrt(fr["Ts0"] ** 2 + fr["Td0"] ** 2) / 1e6,
        label="shear stress",
        color="k",
        linestyle = ls
    )
    ax.plot(
        fr["Time"],
        cohesion + fr["Mud"] * abs(fr["Pn0"] / 1e6),
        label="fault strength",
        color="r",
        linestyle = ls
    )
# plt.plot(fr['Time'], np.sqrt(fr['SRs']**2 + fr['SRd']**2), label='slip rate (m/s)')
# plt.title(f"({fr['x']:.0f}, {fr['y']:.0f}, {fr['z']:.0f})")
# plt.xlim([6,9])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.set_ylabel("stress and strength (MPa)")
ax.set_xlabel("time (s)")
ax.legend()
plt.show()
fn = f"output/stress_strength_at_hypocenter.svg"
fig.savefig(fn, bbox_inches="tight")
print(f"done write {fn}")
