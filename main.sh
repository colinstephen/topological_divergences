#!/usr/bin/env bash

# args to the script are SEED LENGTH SAMPLES MESH
declare -A args
args[0]="42 500 512"
args[3]="42 750 512"
args[4]="42 1000 512"

for arg_set in "${args[@]}"; do
    python main.py $arg_set
done
