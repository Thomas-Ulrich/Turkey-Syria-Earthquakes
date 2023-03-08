#!/bin/bash
set -eo pipefail
prefix=$1
python ~/TuSeisSolScripts/displayh5vtk/displayUnstructuredVtk.py $1-fault.xdmf --Data SR --pvcc pvcc/SEviewTurkey.pvcc --idt -1 --BoundaryEdges --zoom 1.5 --title_scalar_bar "slip rate (m/s)" --crange 0 4 --colorScale nuuk --scalar_bar 0.9 0.2 --winSize 1600 900  --timetext "0.5 black 0.1 0.1" --OSR --oneDtMem --clip "0 60e3 0 -0.7682212795973759 1.0 0"
