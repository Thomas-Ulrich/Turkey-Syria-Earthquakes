#!/bin/bash

myfold=$1

outFile="figureSR.png"
nxfig=1
nyfig=6

function calcgxgy()
{
    gx=$((0+1600*($k0%$nxfig)))
    gy=$((0+$k0/$nxfig*200))
    echo $k0 $gx $gy
    k0=$(($k0+1))
}


#create the background
#3400  = 900 + 5*500
sx=$((000+1600*$nxfig))
sy=$((000+300*$nyfig))

convert -size ${sx}x$sy xc:transparent $outFile
k0=0

for k in 4 10 20 28 40 60 80
do
    python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data SR --pvcc SEviewTurkey_flatter.pvcc --at_time $k --BoundaryEdges --zoom 3.5 --crange 0 4 --colorScale viridis_r0.xml --winSize 1600 450  --OSR --oneDtMem --output trash_me 
   calcgxgy
   convert output/trash_me.png -transparent white output/trash_me.png
   convert $outFile output/trash_me.png -geometry +$gx+$gy -composite $outFile
done
