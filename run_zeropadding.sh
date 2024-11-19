#!/bin/bash

# Default values for parameters
MOLECULE="Cr2"
NOISE=0.1
TMAX=1000
OVERLAP=0.2
DT=1
OPTION="ff=0.2_left_right"
DENOISED="False"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --molecule) MOLECULE="$2"; shift ;;
        --noise) NOISE="$2"; shift ;;
        --Tmax) TMAX="$2"; shift ;;
        --overlap) OVERLAP="$2"; shift ;;
        --dt) DT="$2"; shift ;;
        --option) OPTION="$2"; shift ;;
        --denoised) DENOISED="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done
NUMPAD_VALUES=(512 1024 2048 4096 8192)

# Loop over numpad values and run the script
for numpad in "${NUMPAD_VALUES[@]}"
do
    echo "Running script with numpad=$numpad, molecule=$MOLECULE, noise=$NOISE, Tmax=$TMAX, overlap=$OVERLAP, dt=$DT, option=$OPTION, denoised=$DENOISED"
    python dft_zero_pad.py --molecule "$MOLECULE" --noise "$NOISE" --Tmax "$TMAX" --overlap "$OVERLAP" --dt "$DT" --numpad "$numpad" --option "$OPTION" --denoised "$DENOISED"
done
