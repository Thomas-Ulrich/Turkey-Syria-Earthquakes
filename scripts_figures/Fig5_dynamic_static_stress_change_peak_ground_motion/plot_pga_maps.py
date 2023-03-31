'''
Script to make maps comparing observed/synthetic PGAs for the two events.
'''


import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
from impactutils.mapping.scalebar import draw_scale
import argparse

# parsing python arguments
parser = argparse.ArgumentParser(description="plot map comparing observed/synthetic PGAs or PGVs for both events")
parser.add_argument(
    "--PGV",
    dest="PGV",
    default=False,
    action="store_true",
    help="plot PGV instead of PGA",
)
args = parser.parse_args()


PGA_PGV = "PGV" if args.PGV else "PGA"
# USGS fault traces
# TODO: use fault traces for the actual model setup
shapefile = gpd.read_file(
    'fault_traces/simple_fault_2023-02-13/simple_fault_2023-2-13.shp')


# Function for converting PGAs to plotting circle size
def pga_to_size(pga):
    return 0.05e3 * np.log(pga) - 0.02e3


files = [f'obs_us6000jllz_{PGA_PGV}.csv', f'syn_us6000jllz_{PGA_PGV}.csv',
         f'obs_us6000jlqa_{PGA_PGV}.csv', f'syn_us6000jlqa_{PGA_PGV}.csv']

dfs = [pd.read_csv(file, dtype={"codes": str}) for file in files]

df1 = dfs[0].merge(dfs[1], on='codes')
df2 = dfs[2].merge(dfs[3], on='codes')

# Remove synthetic PGAs for the second event that were too low
# because the simulation wasn't long enough. TODO: remove
# this once using the longer simulations.
df2 = df2[df2.pgas_y > 0.2]

stamen_terrain = cimgt.Stamen('terrain-background')
evlons = [37.019, 37.206]
evlats = [37.220, 38.016]

fig = plt.figure(figsize=(16, 15))
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1, projection=stamen_terrain.crs)

    if i == 0 or i == 2:
        df = df1
    else:
        df = df2

    if i == 0 or i == 1:
        times = df['times_x']
        pgas = df['pgas_x']
    else:
        times = df['times_y']
        pgas = 100 * df['pgas_y']


    # TODO: use diferent colorscale for each event and decrease the
    # minimum time (currently 40 s)
    if i == 0 or i == 2:
        vmin = 40
        vmax = 110
    if i == 1 or i == 3:
        vmin = 40
        vmax = 110

    # Plot PGAs
    im = ax.scatter(
        df['lons_x'], df['lats_x'], c=times,
        transform=ccrs.Geodetic(), vmin=vmin, vmax=vmax, s=pga_to_size(pgas),
        edgecolors='k', lw=0.2)

    # Epicenter
    ax.scatter(evlons[i % 2], evlats[i % 2], marker='*', s=300,
               c='r', transform=ccrs.Geodetic(), label='Epicenter')

    # Stamen terrain
    ax.add_image(stamen_terrain, 8)
    ax.set_extent([34.2, 39, 35.3, 39], crs=ccrs.PlateCarree())

    # Faults
    shapefile.plot(ax=ax, transform=ccrs.Geodetic(), color='k', ls='--')

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.left_labels = True
    if i == 0 or i == 1:
        gl.bottom_labels = False
    if i == 1 or i == 3:
        gl.left_labels = False

    if i == 0:
        for val, lab in zip([981, 98.1, 9.81], ['1 g', '0.1 g', '0.01 g']):
            ax.scatter(
                [], [], s=pga_to_size(val), edgecolors='k', lw=0.5,
                facecolors='none', label=lab)
            plt.legend(labelspacing=1, borderpad=1)

    # Distance scale bar
    if i == 0:
        draw_scale(ax, pos='lr')

# Final adjustments
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.35, 0.02, 0.3])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('PGA time (s after origin time)', fontsize=20)
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.savefig('pga_times_obs_simulated.pdf', bbox_inches='tight')
plt.show()
