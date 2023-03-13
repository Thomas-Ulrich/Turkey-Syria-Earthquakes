#!/bin/bash
set -eo pipefail
# rake figure
#python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Rake180 --pvcc pvcc/SEviewTurkey_flatter_2nd.pvcc --last --BoundaryEdges --zoom 2.0 --crange 180 230 --colorScale hawaii --winSize 1000 500  --OSR --oneDtMem --output rake --clip "0 60e3 0 0.7682212795973759 -1.0 0" --title_scalar_bar "rake" --scalar_bar 0.8 0.35
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data dCFS_Turkey --pvcc pvcc/SEviewTurkey_flatter_2nd.pvcc --last --BoundaryEdges --zoom 2.0 --colorScale roma_r --winSize 1000 500  --OSR --oneDtMem --output dCFS --clip "0 60e3 0 0.7682212795973759 -1.0 0" --crange " -1.5e6" 1.5e6
python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --input output/dCFS.png --output output/dCFS_transparent.png
