#!/bin/bash
set -eo pipefail

python compute_average_Vr.py $1 --time_range 0 100
python compute_average_Vr.py $2 --time_range 100 180
