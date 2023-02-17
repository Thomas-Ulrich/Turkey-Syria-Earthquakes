#!/bin/bash
set -eo pipefail

myfold=$1

outFile="figureSlipVr.png"


python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data ASl --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 3.5 --crange 0 6 --colorScale batlowK_r0 --winSize 1600 450  --OSR --oneDtMem --output trash_me1

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Vr --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 3.5 --crange 0 5000 --colorScale lajolla0 --winSize 1600 450  --OSR --oneDtMem --output trash_me2

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Sld --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 3.5 --crange -1 1 --colorScale bam --winSize 1600 450  --OSR --oneDtMem --output trash_me3

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs output/trash_me1.png output/trash_me2.png output/trash_me3.png  --rel 0.8 --output $outFile
