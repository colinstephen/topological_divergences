#!/usr/bin/env bash

# args to the script are SEED LENGTH SAMPLES MESH
declare -A args
args[0]="42 500 1000 0.75"
args[1]="42 500 1000 0.5"
args[2]="42 500 1000 0.25"
args[3]="42 1000 1000 0.75"
args[4]="42 1000 1000 0.5"
args[5]="42 1000 1000 0.25"

for arg_set in "${args[@]}"; do
    python experiments5.py $arg_set
done
