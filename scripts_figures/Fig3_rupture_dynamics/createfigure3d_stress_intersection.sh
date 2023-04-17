#!/bin/bash
myfold=$1

outFile="figure_tractions_intersection.png"
filelist=""
for k in 12 15.0
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data T_s --pvcc ../pvcc/focus_intersection.pvcc --at_time $k --BoundaryEdges --zoom 5.0 --crange " -1e7" 1e7 --colorScale roma --winSize 800 800  --OSR --oneDtMem --output trash_me_Ts$k --clip "0 60e3 0 -0.7682212795973759 1.0 0"
   filelist="$filelist output/trash_me_Ts$k.png"
done
for k in 12 15.0
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data P_n --pvcc ../pvcc/focus_intersection.pvcc --at_time $k --BoundaryEdges --zoom 5.0 --crange " -2e7" 2e7 --colorScale vik --winSize 800 800  --OSR --oneDtMem --output trash_me_Pn$k --clip "0 60e3 0 -0.7682212795973759 1.0 0"
   filelist="$filelist output/trash_me_Pn$k.png"
done

echo $filelist

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs $filelist  --rel 0.6 1.0 --col 2 --output $outFile
