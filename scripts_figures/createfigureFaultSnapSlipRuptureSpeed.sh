#!/bin/bash

myfold=$1

outFile="figureSlipVr.png"
nxfig=1
nyfig=3

function calcgxgy()
{
    gx=$((0+1600*($k0%$nxfig)))
    gy=$((0+$k0/$nxfig*400))
    echo $k0 $gx $gy
    k0=$(($k0+1))
}
#create the background
#3400  = 900 + 5*500
sx=$((000+1600*$nxfig))
sy=$((000+450*$nyfig))

convert -size ${sx}x$sy xc:transparent $outFile
k0=0

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data ASl --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 3.5 --crange 0 5 --colorScale batlowK_r0 --winSize 1600 450  --OSR --oneDtMem --output trash_me 
calcgxgy
convert output/trash_me.png -transparent white output/trash_me.png
convert $outFile output/trash_me.png -geometry +$gx+$gy -composite $outFile

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Vr --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 3.5 --crange 0 5000 --colorScale lajolla0 --winSize 1600 450  --OSR --oneDtMem --output trash_me
calcgxgy
convert output/trash_me.png -transparent white output/trash_me.png
convert $outFile output/trash_me.png -geometry +$gx+$gy -composite $outFile

python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data Sld --pvcc pvcc/SEviewTurkey_flatter.pvcc --last --BoundaryEdges --zoom 3.5 --crange -1 1 --colorScale bam --winSize 1600 450  --OSR --oneDtMem --output trash_me
calcgxgy
convert output/trash_me.png -transparent white output/trash_me.png
convert $outFile output/trash_me.png -geometry +$gx+$gy -composite $outFile
