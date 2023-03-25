import pandas as pd
from pyproj import Transformer
import numpy as np
import matplotlib.pylab as plt
import trimesh
import seissolxdmf
import argparse
from scipy import spatial


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def compute_centers(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        return (xyz[connect[:, 0]] + xyz[connect[:, 1]] + xyz[connect[:, 2]]) / 3.0

    def compute_strike(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        strike = np.zeros((sx.nElements, 2))

        for i in range(sx.nElements):
            a = xyz[connect[i, 1]] - xyz[connect[i, 0]]
            b = xyz[connect[i, 2]] - xyz[connect[i, 0]]
            if (a[0] ** 2 + a[1] ** 2) > (b[0] ** 2 + b[1] ** 2):
                strike[i, :] = a[:2]
            else:
                strike[i, :] = b[:2]
        strike = strike[:, :] / np.linalg.norm(strike[:, :], axis=1)[:, None]
        return strike


def get_fault_trace():
    fn = args.fault[0]
    sx = seissolxdmf.seissolxdmf(fn)
    geom = sx.ReadGeometry()
    connect = sx.ReadConnect()
    mesh = trimesh.Trimesh(geom, connect)
    # list vertex of the face boundary
    unique_edges = mesh.edges[
        trimesh.grouping.group_rows(mesh.edges_sorted, require_count=2)
    ]
    unique_edges = unique_edges[:, :, 1]
    ids_external_nodes = np.unique(unique_edges.flatten())

    nodes = mesh.vertices[ids_external_nodes, :]
    nodes = nodes[nodes[:, 2] > 0]
    nodes = nodes[nodes[:, 1].argsort()]

    # Compute strike vector to filter boundaries of near-vertical edges
    grad = np.gradient(nodes, axis=0)
    grad = grad / np.linalg.norm(grad, axis=1)[:, None]

    ids_top_trace = np.where(np.abs(grad[:, 2]) < 0.8)[0]
    nodes = nodes[ids_top_trace]
    return nodes


# parsing python arguments
parser = argparse.ArgumentParser(description="extract slip profile along fault trace")
parser.add_argument(
    "--event",
    nargs=1,
    help="1: mainshock, 2: aftershock",
    choices=[1, 2],
    required=True,
    type=int,
)
parser.add_argument("--fault", nargs="+", help="fault xdmf file name", required=True)
parser.add_argument(
    "--downsample", nargs=1, help="take one node every n", default=[1], type=int
)
args = parser.parse_args()


plt.rc("font", family="FreeSans", size=12)
# plt.rc("font", size=8)
# plt.rcParams["text.usetex"] = True

if args.event[0] == 1:
    fn = "../../ThirdParty/offset_sentinel2.txt"
    event = "mainshock"
else:
    fn = "../../ThirdParty/EW_offset_sentinel2_Mw75.txt"
    event = "2nd"

df = pd.read_csv(fn, sep=" ")
df = df.sort_values(by=["lon", "lat"])


myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)
x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
xy = np.vstack((x, y)).T
dist = np.linalg.norm(xy[1:, :] - xy[0:-1, :], axis=1)
acc_dist = np.add.accumulate(dist) / 1e3
acc_dist = np.insert(acc_dist, 0, 0)

trace_nodes = get_fault_trace()[:: args.downsample[0]]

fig = plt.figure(figsize=(5.5, 3.0))
ax = fig.add_subplot(111)
ax.set_xlabel("distance along strike (km)")
ax.set_ylabel("fault offset (m)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

lw = 0.5
if args.event[0] == 1:
    plt.errorbar(
        acc_dist,
        df["ns_offset"],
        yerr=df["ns_error"],
        color="b",
        linestyle="--",
        linewidth=lw,
        label="NS from Sentinel 2",
    )
plt.errorbar(
    acc_dist,
    df["ew_offset"],
    yerr=df["ew_error"],
    color="orange",
    linestyle="--",
    linewidth=lw,
    label="EW from Sentinel 2",
)

for i, fn in enumerate(args.fault):
    sx = seissolxdmfExtended(fn)
    fault_centers = sx.compute_centers()
    strike = sx.compute_strike()

    idt = sx.ReadNdt() - 1
    Sls = np.abs(sx.ReadData("Sls", idt))

    tree = spatial.KDTree(fault_centers)
    dist, idsf = tree.query(trace_nodes)

    slip_at_trace = Sls[idsf]
    strike = strike[idsf]

    tree = spatial.KDTree(trace_nodes[:, 0:2])
    dist, idsf2 = tree.query(xy)
    slip_at_trace = slip_at_trace[idsf2]
    strike = strike[idsf2]

    ew = np.abs(slip_at_trace * strike[:, 0])
    ns = np.abs(slip_at_trace * strike[:, 1])

    plt.plot(acc_dist, ns, "b", linewidth=lw * (1 + 0.5 * i), label="NS synthetics")
    plt.plot(
        acc_dist, ew, "orange", linewidth=lw * (1 + 0.5 * i), label="EW synthetics"
    )
    if args.event[0] == 1:
        ids = df["ns_offset"].notna()
        Chi2_ns = np.sum(
            (ns[ids] - df["ns_offset"][ids].to_numpy()) ** 2
            / df["ns_error"][ids].to_numpy() ** 2
        )
    else:
        Chi2_ns = 0.0
    ids = df["ew_offset"].notna()
    Chi2_ew = np.sum(
        (ew[ids] - df["ew_offset"][ids].to_numpy()) ** 2
        / df["ew_error"][ids].to_numpy() ** 2
    )
    Chi = Chi2_ew + Chi2_ns
    print(f"{fn}: (Chi2 Chi2_ew Chi2_ns) = {Chi} {Chi2_ew} {Chi2_ns}")

if args.event[0] == 1:
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(0.1, 1.05), loc="center left")

if not os.path.exists("output"):
    os.makedirs("output")
fn = f"output/comparison_offset_{event}.svg"
plt.savefig(fn, dpi=200)
print(f"done writing {fn}")
