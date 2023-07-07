import numpy as np
import argparse
import seissolxdmf
import seissolxdmfwriter as sxw


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def OutputTimes(self):
        """returns the list of output times written in the file"""
        root = self.tree.getroot()
        outputTimes = []
        for Property in root.findall("Domain/Grid/Grid/Time"):
            outputTimes.append(float(Property.get("Value")))
        return outputTimes

def compute_time_indices(sx, at_time):
    """retrive list of time index in file"""
    outputTimes = np.array(sx.OutputTimes())
    lidt = []
    for oTime in at_time:
        idsClose = np.where(np.isclose(outputTimes, oTime, atol=0.0001))
        if not len(idsClose[0]):
            print(f"t={oTime} not found in {sx.xdmfFilename}")
        else:
            lidt.append(idsClose[0][0])
    return lidt

parser = argparse.ArgumentParser(description="compute max dynanmic dCFS")
parser.add_argument("xdmf_filename", help="seissol xdmf file")
parser.add_argument(
    "--at_time",
    nargs="+",
    help="list of output time values to vizualize (cannot be combined with idt/vidt)",
    type=float,
)

args = parser.parse_args()


sx = seissolxdmfExtended(args.xdmf_filename)
# Compute dCFS with shear traction in the rake direction of the final slip
lidt = compute_time_indices(sx, args.at_time)
print(lidt)
ASl = np.zeros((len(lidt),sx.nElements))
dt = sx.ReadTimeStep()

for k, i in enumerate(lidt[1:]):
    ASl[k,:] = sx.ReadData("ASl", i) - sx.ReadData("ASl", i-1)
print(ASl)
print(args.at_time[1:], lidt[1:])
sxw.write_seissol_output(
    "ASl_acc",
    sx.ReadGeometry(),
    sx.ReadConnect(),
    ["ASl_acc"],
    [ASl],
    1.0,
    range(len(lidt)-1),
)
