#!/bin/bash
set -eo pipefail

python compare_geodetic.py --band EW --surface $1-surface.xdmf --down 4 --extension svg --no --vmax 6
python compare_geodetic.py --band NS --surface $1-surface.xdmf --down 4 --extension svg --no
python compare_geodetic.py --band azimuth --surface $1-surface.xdmf --down 6 --extension svg --no
python compare_geodetic.py --band range --surface $1-surface.xdmf --down 6 --extension svg --no
python compare_geodetic.py --band los77 --surface $1-surface.xdmf --down 6 --extension png --no 
python compare_geodetic.py --band los184 --surface $1-surface.xdmf --down 6 --extension png --no 
#python compare_offset_S2.py --event 1 --fault $1-fault.xdmf
#python compare_offset_S2.py --event 2 --fault $1-fault.xdmf

