import numpy as np
import seissolxdmf
import argparse


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def compute_centers(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        return (xyz[connect[:, 0]] + xyz[connect[:, 1]] + xyz[connect[:, 2]]) / 3.0


# parsing python arguments
parser = argparse.ArgumentParser(description="compute median Vr")
parser.add_argument(
    "--time_range",
    nargs=2,
    help="time considered for computing median Vr",
    required=True,
    type=float,
)
parser.add_argument("fault", help="fault xdmf file name")
args = parser.parse_args()
print(f"using time range, {args.time_range}")

fn = args.fault
sx = seissolxdmfExtended(fn)
xyz = sx.compute_centers()
ndt = sx.ReadNdt() - 1
Vr = sx.ReadData("Vr", ndt)
RT = sx.ReadData("RT", ndt)
ASl = sx.ReadData("ASl", ndt)
ids0 = np.where(RT > args.time_range[0])[0]
ids1 = np.where(RT < args.time_range[1])[0]
ids2 = np.where(ASl > 0.1)[0]
ids = np.intersect1d(ids0, ids1)
ids = np.intersect1d(ids, ids2)
print("median event", np.median(Vr[ids]))

if args.time_range[0] == 0.0:
    ids_loc = np.where(xyz[:, 0] > 22e3)[0]
    ids2 = np.intersect1d(ids, ids_loc)
    print("1st event, East", np.median(Vr[ids2]))
    ids_loc = np.where(xyz[:, 0] < -9e3)[0]
    ids2 = np.intersect1d(ids, ids_loc)
    print("1st event, West", np.median(Vr[ids2]))
else:
    ids_loc = np.where(xyz[:, 0] > 20e3)[0]
    ids2 = np.intersect1d(ids, ids_loc)
    print("2nd event, East", np.median(Vr[ids2]))
    ids_loc = np.where(xyz[:, 0] < 20e3)[0]
    ids2 = np.intersect1d(ids, ids_loc)
    print("2nd event, West", np.median(Vr[ids2]))
