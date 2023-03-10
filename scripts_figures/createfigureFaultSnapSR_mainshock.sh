#!/bin/bash
myfold=$1

outFile="figureSR_mainshock.png"
filelist=""
#for k in 4 10 14 20 28 52 68
for k in 8 14 18 32 52 68
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data SR --pvcc pvcc/SEviewTurkey_flatter.pvcc --at_time $k --BoundaryEdges --zoom 3.3 --crange 0 8 --colorScale viridis_r0 --winSize 1600 450  --OSR --oneDtMem --output trash_me$k --opa 0.99 --clip "0 60e3 0 -0.7682212795973759 1.0 0"
   filelist="$filelist output/trash_me$k.png"
done
echo $filelist

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs $filelist  --rel 0.35 1.0 --output $outFile
