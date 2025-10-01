#!/bin/bash -l

set -e

## Log file
log="./log_compsep_example"

basedir=  ## YOUR CURLICAST DIR

cd $basedir
config=examples/example_paramfile.yaml


echo "Launching pipeline at $(date)"
echo "Logging to ${log}"

python -u curlicast/compsep_nopipe.py --config $config > ${log} 2>&1
python -u curlicast/plotter_nopipe.py --config $config > ${log} 2>&1

echo "Ending batch script at $(date)"