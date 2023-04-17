#!/bin/bash
set -eo pipefail

# For figure 2b
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py viridis_r --crange 0 8 --labelfont 8 --hor --height 1.2 3.6
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py hawaii_r --crange 0 10 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py bam --crange -4 4 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py davos --crange 0 9 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py buda --crange 0 6000 --labelfont 8 --hor --height 1.2 3.6 --nticks 3

# For the tractions plot (Fig 2d)
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py roma --crange -1 1 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py vik --crange -2 2 --labelfont 8 --hor --height 1.2 3.6 --nticks 3

