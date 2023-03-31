import numpy as np

ref_region_theta = np.radians(45)
ct = np.cos(ref_region_theta)
st = np.sin(ref_region_theta)
u = np.array([ct, st])
v = np.array([-st, ct])
xc = np.array([20e3, 50e3])
xyz = np.zeros((4, 3))
ik = 0
for i, j in [[1, 1], [-1, 1], [-1, -1], [1, -1]]:
    xyz[ik, 0:2] = xc + i * 200e3 * u + j * 100e3 * v
    xyz[ik, 2] = 2e3
    ik = ik + 1

nvert = xyz.shape[0]
segments = np.zeros((4, 2))
segments[0, :] = [0, 1]
segments[1, :] = [1, 2]
segments[2, :] = [2, 3]
segments[3, :] = [3, 0]
nseg = segments.shape[0]

# Now write vtk file
with open("RefinedArea.vtk", "w") as fout:
    fout.write(
        f"""# vtk DataFile Version 2.0
parabola - polyline
ASCII
DATASET POLYDATA
POINTS {nvert} float
"""
    )
    np.savetxt(fout, xyz, "%e %e %e")
    fout.write(f"\nLINES {nseg} {3*nseg}\n")
    np.savetxt(fout, segments, "2 %d %d")
    fout.write("\n")

print("RefinedArea.vtk successfully created")
