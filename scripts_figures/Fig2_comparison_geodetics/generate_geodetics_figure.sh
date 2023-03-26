#!/bin/bash
set -eo pipefail

if [ -z "$2" ]
then
ext=svg
else
ext=$2
fi

if [ -z "$3" ]
then
diff_or_not=
else
diff_or_not="--diff"
fi


python compare_geodetic.py --band EW --surface $1-surface.xdmf --down 4 --extension $ext --no --vmax 6 $diff_or_not
python compare_geodetic.py --band NS --surface $1-surface.xdmf --down 4 --extension $ext --no $diff_or_not
python compare_geodetic.py --band azimuth --surface $1-surface.xdmf --down 6 --extension $ext --no $diff_or_not
python compare_geodetic.py --band range --surface $1-surface.xdmf --down 6 --extension $ext --no $diff_or_not
python compare_geodetic.py --band los77 --surface $1-surface.xdmf --down 6 --extension $ext --no $diff_or_not
python compare_geodetic.py --band los184 --surface $1-surface.xdmf --down 6 --extension $ext --no $diff_or_not
python compare_offset_S2.py --event 1 --fault $1-fault.xdmf
python compare_offset_S2.py --event 2 --fault $1-fault.xdmf

