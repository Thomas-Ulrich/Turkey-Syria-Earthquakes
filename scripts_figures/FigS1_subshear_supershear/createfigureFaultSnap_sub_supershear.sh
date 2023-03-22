#!/bin/bash

python moment_rate.py $2 $1 --label supershear subshear

outFile="figureSR_mainshock_sub_supershear.png"
filelist=""
for k in 2 4 10 12 14 16
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1SR-fault.xdmf --Data SR --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --at_time $k --BoundaryEdges --zoom 5.0 --crange 0 8 --colorScale viridis_r0 --winSize 1000 600  --OSR --oneDtMem --output trash_me$k --opa 0.99 --clip "0 60e3 0 -0.7682212795973759 1.0 0"
   filelist="$filelist output/trash_me$k.png"
done
for k in 2 4 10 12 14 16
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $2SR-fault.xdmf --Data SR --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --at_time $k --BoundaryEdges --zoom 5.0 --crange 0 8 --colorScale viridis_r0 --winSize 1000 600  --OSR --oneDtMem --output trash_me_sup_$k --opa 0.99 --clip "0 60e3 0 -0.7682212795973759 1.0 0"
   filelist="$filelist output/trash_me_sup_$k.png"
done

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs $filelist  --rel 0.5 1.05 --output $outFile --col 2
 
outFile="figureSlipVr_mainshock_sub_supershear.png"
   
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1last-fault.xdmf --Data ASl --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --BoundaryEdges --zoom 5.0 --crange 0 10 --colorScale hawaii_r0 --winSize 1000 600  --OSR --oneDtMem --output trash_me1 --clip "0 60e3 0 -0.7682212795973759 1.0 0" --last
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1last-fault.xdmf --Data Vr --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --BoundaryEdges --zoom 5.0 --crange 0 6000 --colorScale buda0 --winSize 1000 600  --OSR --oneDtMem --output trash_me2 --clip "0 60e3 0 -0.7682212795973759 1.0 0" --last
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $2last-fault.xdmf --Data ASl --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --BoundaryEdges --zoom 5.0 --crange 0 10 --colorScale hawaii_r0 --winSize 1000 600  --OSR --oneDtMem --output trash_me_sup1 --clip "0 60e3 0 -0.7682212795973759 1.0 0" --last
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $2last-fault.xdmf --Data Vr --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --BoundaryEdges --zoom 5.0 --crange 0 6000 --colorScale buda0 --winSize 1000 600  --OSR --oneDtMem --output trash_me_sup2 --clip "0 60e3 0 -0.7682212795973759 1.0 0" --last
python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs output/trash_me1.png output/trash_me2.png output/trash_me_sup1.png output/trash_me_sup2.png  --rel 0.5 1.05 --output $outFile --col 2
