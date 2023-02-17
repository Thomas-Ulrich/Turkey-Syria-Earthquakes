#!/bin/bash
myfold=$1

outFile="figureSR.png"
filelist=""
for k in 4 10 20 28 40 60 80
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data SR --pvcc pvcc/SEviewTurkey_flatter.pvcc --at_time $k --BoundaryEdges --zoom 3.5 --crange 0 4 --colorScale viridis_r0 --winSize 1600 450  --OSR --oneDtMem --output trash_me$k --opa 0.99
   filelist="$filelist output/trash_me$k.png"
done
echo $filelist

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs $filelist  --rel 0.35 --output $outFile
