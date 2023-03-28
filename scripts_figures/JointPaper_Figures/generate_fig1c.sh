#!/bin/bash
set -evo pipefail
python JointPaper_fig1C.py --event 1  --fault $1-fault.xdmf
