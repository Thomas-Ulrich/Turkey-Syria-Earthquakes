#!/bin/bash
myfold=$1

outFile="figureS4$2.png"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data ASl --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange 0 10 --colorScale hawaii_r0 --winSize 1600 550  --OSR --oneDtMem --output trash_me1 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Sld --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange -4 4 --colorScale bam --winSize 1600 550  --OSR --oneDtMem --output trash_me2 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data PSR --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange 0 9 --colorScale davos0 --winSize 1600 550  --OSR --oneDtMem --output trash_me3 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Vr --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 1.0 --crange 0 6000 --colorScale buda0 --winSize 1600 550  --OSR --oneDtMem --output trash_me4 --clip "0 60e3 0 0.7682212795973759 -1.0 0"

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs output/trash_me1.png output/trash_me2.png output/trash_me3.png output/trash_me4.png  --rel 0.3 1.0 --output $outFile
