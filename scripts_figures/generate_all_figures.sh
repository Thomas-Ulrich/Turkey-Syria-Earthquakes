#!/bin/bash
set -eo pipefail

geodetic_figure=0
colorbars=1
sub_supershear_figure=0
#python moment_rate.py $1 --label 'dynamic rupture scenario'

if [ $colorbars -eq 1 ]
then
# For figure 1a to c
#python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py viridis --crange 0 8 --labelfont 8 --hor --height 1.2 3.6
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py hawaii_r --crange 0 10 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py buda --crange 0 6000 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py davos --crange 0 9 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
#python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py bam --crange -4 4 --labelfont 8 --hor --height 1.2 3.6 --nticks 3

# For the tractions plot (Fig 1d)
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py roma --crange -1 1 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py roma_r --crange -1.5 1.5 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py vik --crange -2 2 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
fi

#./createfigureFaultSnapSR.sh $1

if [ $geodetic_figure -eq 1 ]
then
python compare_geodetic.py --band EW --surface $1-surface.xdmf --down 4 --extension svg --no
python compare_geodetic.py --band NS --surface $1-surface.xdmf --down 4 --extension svg --no
python compare_geodetic.py --band azimuth --surface $1-surface.xdmf --down 6 --extension svg --no
python compare_geodetic.py --band range --surface $1-surface.xdmf --down 6 --extension svg --no
python compare_offset_S2.py --event 1 --fault $1-fault.xdmf
python compare_offset_S2.py --event 2 --fault $1-fault.xdmf
fi

if [ $sub_supershear_figure -eq 1 ]
then
./createfigureFaultSnapSR_sub_supershear.sh $1 $2
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py viridis --crange 0 8 --labelfont 12 --hor --height 1.2 3.6
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py hawaii_r --crange 0 10 --labelfont 12 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py buda --crange 0 6000 --labelfont 12 --hor --height 1.2 3.6 --nticks 3
fi
