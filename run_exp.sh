#!/bin/bash

source activate py27
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="$3"

PYTHONPATH=./:$PYTHONPATH python morphodecomp.py $1 $2
