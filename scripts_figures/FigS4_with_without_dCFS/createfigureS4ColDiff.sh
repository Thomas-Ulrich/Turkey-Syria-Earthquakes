#!/bin/bash
myfold=$1

outFile="figureS4$2.png"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data diff_ASl --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange -2 2 --colorScale vik --winSize 1600 550  --OSR --oneDtMem --output trash_me1 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data diff_Sld --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange -2 2 --colorScale vik --winSize 1600 550  --OSR --oneDtMem --output trash_me2 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data diff_PSR --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange -2 2 --colorScale vik --winSize 1600 550  --OSR --oneDtMem --output trash_me3 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data diff_Vr --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange -2000 2000 --colorScale vik --winSize 1600 550  --OSR --oneDtMem --output trash_me4 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs output/trash_me1.png output/trash_me2.png output/trash_me3.png output/trash_me4.png  --rel 0.3 1.0 --output $outFile
