#!/bin/bash
set -eo pipefail

python moment_rate.py $1 --label 'dynamic rupture scenario'  --t0_2nd 150.
