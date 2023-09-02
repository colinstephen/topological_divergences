#!/usr/bin/env bash

# args to the script are SEED LENGTH SAMPLES MESH
declare -A args
# args[0]="42 500 500"
# args[1]="42 750 500"
# args[2]="42 1000 500"
# args[0]="42 600 500"
args[0]="42 750 500"
# args[1]="42 1000 500"

for arg_set in "${args[@]}"; do
    python main.py $arg_set
done
