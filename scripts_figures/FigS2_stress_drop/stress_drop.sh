#!/bin/bash
myfold=$1

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data stress_drop_Turkey --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 2.7 --crange 0 1.7e7  --colorScale roma_r0 --winSize 1600 550  --OSR --oneDtMem --output trash_stress_drop
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py roma_r --crange 0 17 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
