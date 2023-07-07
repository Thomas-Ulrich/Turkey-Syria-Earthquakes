#!/bin/bash
set -eo pipefail
prefix=$1
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data SR --pvcc ../pvcc/SEviewTurkey.pvcc --idt -1 --BoundaryEdges --zoom 1.5 --title_scalar_bar "slip rate (m/s)" --crange 0 4 --colorScale viridis_r0 --scalar_bar 0.9 0.2 --winSize 1600 900  --timetext "0.1 black 0.1 0.1" --OSR --oneDtMem --clip "0 60e3 0 -0.7682212795973759 1.0 0"
prefix_underscores=${1//\//_}
ffmpeg -framerate 40 -i output/${prefix_underscores}-fault_SEviewTurkey_SR_%d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 -vcodec libx264 -pix_fmt yuv420p  -q:v 1 mainshock.mp4
