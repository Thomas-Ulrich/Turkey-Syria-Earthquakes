#!/bin/bash
set -eov pipefail
python moment_rate.py $1 --label 'dynamic rupture scenario'
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py viridis --crange 0 4 --labelfont 8 --hor --height 1.2 3.6
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py batlowK_r --crange 0 6 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py lajolla --crange 0 5000 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py bam --crange -1 1 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
#./createfigureFaultSnapSR.sh $1
