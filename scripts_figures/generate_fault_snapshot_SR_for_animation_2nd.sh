#!/bin/bash
set -eo pipefail
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data SR --pvcc pvcc/SEviewTurkey_flatter_2nd.pvcc --idt $(seq 200 1 280) --BoundaryEdges --zoom 2.0 --crange 0 4 --colorScale viridis_r0 --winSize 1000 500  --OSR --oneDtMem --opa 0.99 --clip "0 60e3 0 0.7682212795973759 -1.0 0" --title_scalar_bar "slip rate (m/s)" --scalar_bar 0.8 0.35 --timetext "0.5 black 0.1 0.1" 
prefix_underscores=${1//\//_}
ffmpeg -framerate 15 -start_number 200 -i output/${prefix_underscores}-fault_SEviewTurkey_flatter_2nd_SR_%d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 -vcodec libx264 -pix_fmt yuv420p  -q:v 1 2nd_event.mp4
echo "done writing 2nd_event.mp4"
rm output/${prefix_underscores}-fault_SEviewTurkey_flatter_2nd_SR_*
