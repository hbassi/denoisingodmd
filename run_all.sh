#!/bin/bash

MOLECULE="Cr2"
NOISE=0.1
OVERLAP=0.2
DT=1
NUM_TRAJS=1

for Tmax in {15..1000}
do
    echo "Running with Tmax=$Tmax"
    python generate_data.py --molecule $MOLECULE --noise $NOISE --Tmax $Tmax --overlap $OVERLAP --dt $DT --num_trajs $NUM_TRAJS
done
