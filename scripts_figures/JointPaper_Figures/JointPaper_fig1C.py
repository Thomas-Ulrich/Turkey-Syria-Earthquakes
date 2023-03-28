import pandas as pd
from pyproj import Transformer
import numpy as np
import matplotlib.pylab as plt
import trimesh
import seissolxdmf
import argparse
from scipy import spatial
import os


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


plt.rc("font", family="FreeSans", size=8)
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
dist_xy = np.linalg.norm(xy[1:, :] - xy[0:-1, :], axis=1)
acc_dist = np.add.accumulate(dist_xy) / 1e3
acc_dist = np.insert(acc_dist, 0, 0) - 165.0

trace_nodes = get_fault_trace()[:: args.downsample[0]]

fig = plt.figure(figsize=(5.5, 3.0))
ax = fig.add_subplot(111)
ax.set_xlabel("distance along strike (km)")
ax.set_ylabel("rupture time (s)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

lw = 0.8

for i, fn in enumerate(args.fault):
    sx = seissolxdmfExtended(fn)
    fault_centers = sx.compute_centers()
    strike = sx.compute_strike()

    idt = sx.ReadNdt() - 1
    RT = np.abs(sx.ReadData("RT", idt))

    tree = spatial.KDTree(fault_centers)
    dist, idsf = tree.query(trace_nodes)
    strike = strike[idsf]
   
    tree = spatial.KDTree(trace_nodes[:, 0:2])
    dist, idsf2 = tree.query(xy)
    strike = strike[idsf2]
    RT_med = np.zeros_like(strike[:,0])
    RTm2sigma = np.zeros_like(strike[:,0])
    RTp2sigma = np.zeros_like(strike[:,0])
    for ki, xyi in enumerate(xy):
        dis1 = dist_xy[ki-1] if ki-1>0 else dist_xy[ki] 
        dis2 = dist_xy[ki] if ki<len(dist_xy) else dist_xy[ki-1] 
        x0 = np.dot(xyi, strike[ki])
        ud = np.array([strike[ki,1], - strike[ki,0]])
        y0 = np.dot(xyi, ud)
        a1 = x0 - dis1 
        a2 = x0 + dis2
        b1 = y0 - 1e3 
        b2 = y0 + 1e3
        fault_centers_u1 = np.dot(fault_centers[:,0:2], strike[ki])
        id1 = np.where(fault_centers_u1>a1)[0]
        id2 = np.where(fault_centers_u1<=a2)[0]
        id1 = np.intersect1d(id1, id2)
        # remove points not on the current fault
        fault_centers_v1 = np.dot(fault_centers[:,0:2], ud)
        id2 = np.where(fault_centers_v1>b1)[0]
        id1 = np.intersect1d(id1, id2)
        id2 = np.where(fault_centers_v1<=b2)[0]
        id1 = np.intersect1d(id1, id2)
        id2 = np.where(RT>0)[0]
        id1 = np.intersect1d(id1, id2)
        RT_med[ki] = np.median(RT[id1])
        RTm2sigma[ki] = np.percentile(RT[id1], 5)
        RTp2sigma[ki] = np.percentile(RT[id1], 95)

    ax.plot(
        acc_dist,
        RT_med,
        "royalblue",
        linewidth=lw * (1 + 0.5 * i),
        label="Predicted NS offset",
    )
    ax.plot(
        acc_dist,
        RTm2sigma,
        "royalblue",
        linewidth=lw * (1 + 0.5 * i),
        label="Predicted NS offset",
        linestyle=':'
    )
    ax.plot(
        acc_dist,
        RTp2sigma,
        "royalblue",
        linewidth=lw * (1 + 0.5 * i),
        label="Predicted NS offset",
        linestyle=':'
    )

if not os.path.exists("output"):
    os.makedirs("output")
fn = f"output/RT_median_along_strike.pdf"
plt.savefig(fn, dpi=200, bbox_inches="tight")
print(f"done writing {fn}")

df['along_strike_distance'] = acc_dist.tolist()
df['median_rupture_time'] = RT_med.tolist()
df['5pc_rupture_time'] = RTm2sigma.tolist()
df['95pc_rupture_time'] = RTp2sigma.tolist()
df = df.drop(['ew_offset', 'ns_offset','ew_error','ns_error'], axis=1)

fn = f"output/RT_median_along_strike.csv"
df.to_csv(fn)
print(f"done writing {fn}")

