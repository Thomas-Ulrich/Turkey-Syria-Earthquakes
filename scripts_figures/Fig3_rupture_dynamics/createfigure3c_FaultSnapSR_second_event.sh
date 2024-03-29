#!/bin/bash
myfold=$1

outFile="figureSR_second_event.png"
filelist=""
#for k in 2 8 16 20 24 28
for k in 152 158 166 170 174 178
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data SR --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --at_time $k --BoundaryEdges --zoom 1.0 --crange 0 8 --colorScale viridis_r0 --winSize 1600 450  --OSR --oneDtMem --output trash_me$k --opa 0.99 --clip "0 60e3 0 0.7682212795973759 -1.0 0"
   filelist="$filelist output/trash_me$k.png"
done
python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs $filelist  --rel 0.2 1.0 --output $outFile
