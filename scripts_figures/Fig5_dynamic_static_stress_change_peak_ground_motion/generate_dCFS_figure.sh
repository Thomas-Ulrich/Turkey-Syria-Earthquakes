#!/bin/bash
set -eo pipefail

# Dynamic dCFS
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1.xdmf --Data dyn_dCFS --pvcc ../pvcc/SEviewTurkey_flatter_2nd.pvcc --last --BoundaryEdges --zoom 2.0 --colorScale lajolla --winSize 1000 500  --OSR --oneDtMem --output dCFS_dyn --clip "0 60e3 0 0.7682212795973759 -1.0 0" --crange 0 7e6
python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --input output/dCFS_dyn.png --output output/dCFS_dyn_transparent.png
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py lajolla --crange 0 7 --labelfont 8 --hor --height 1.2 3.6 --nticks 3

# Static dCFS
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data dCFS_Turkey --pvcc ../pvcc/SEviewTurkey_flatter_2nd.pvcc --last --BoundaryEdges --zoom 2.0 --colorScale roma_r --winSize 1000 500  --OSR --oneDtMem --output dCFS --clip "0 60e3 0 0.7682212795973759 -1.0 0" --crange " -1.5e6" 1.5e6
python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --input output/dCFS.png --output output/dCFS_transparent.png
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py roma_r --crange -1.5 1.5 --labelfont 8 --hor --height 1.2 3.6 --nticks 3


