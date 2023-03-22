#!/bin/bash
set -evo pipefail

myfile=Turkey78_75_dip70_3_bc-1.xdmf

outFile="SHmax_R0_both.png"


python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $myfile --Data SHmax_Turkey --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --zoom 2.7 --crange -20 70 --colorScale viridis --winSize 1600 550  --OSR --output trash_me1 --lig 0.1 0.6 0.4

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $myfile --Data Dc_Turkey --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --zoom 2.7 --crange 0.5 1.0 --colorScale PuRd --winSize 1600 550  --OSR --output trash_me2 --lig 0.1 0.6 0.4

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $myfile --Data R0_Turkey --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --last --zoom 2.7 --crange 0.0 0.8 --colorScale buda_r --winSize 1600 550  --OSR --output trash_me3 --lig 0.1 0.6 0.4

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1 --Data R_Turkey --pvcc ../pvcc/SEviewTurkey_flatter.pvcc --idt 0 --zoom 2.7 --crange 0.0 0.8 --colorScale buda_r --winSize 1600 550  --OSR --output trash_me4 --lig 0.1 0.6 0.4

python ~/TuSeisSolScripts/displayh5vtk/combine_images_vertically.py --inputs output/trash_me1.png output/trash_me2.png output/trash_me3.png output/trash_me4.png  --rel 0.6 1.0 --col 2 --output $outFile

# Plot colorbars
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py viridis --crange -20 70 --labelfont 8 --hor --height 1.2 3.6 --nticks 3
python ~/TuSeisSolScripts/displayh5vtk/plotColorBar.py buda_r --crange 0.0 0.8 --labelfont 8 --hor --height 1.2 3.6 --nticks 3

