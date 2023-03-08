#!/bin/bash
set -eo pipefail

myfold=$1

outFile="figureSlipVr_both.png"


python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data ASl --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 2.7 --crange 0 10 --colorScale hawaii_r0 --winSize 1600 550  --OSR --oneDtMem --output trash_me1

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Sld --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 2.7 --crange -4 4 --colorScale bam --winSize 1600 550  --OSR --oneDtMem --output trash_me2

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Vr --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 2.7 --crange 0 6000 --colorScale buda0 --winSize 1600 550  --OSR --oneDtMem --output trash_me3

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs output/trash_me1.png output/trash_me2.png output/trash_me3.png  --rel 0.7 --output $outFile
