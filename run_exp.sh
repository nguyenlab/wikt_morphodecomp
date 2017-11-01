#!/bin/bash

source activate py27
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

if [ -z ${3+x} ]; 
then 
    echo "Select GPU number";
    echo "$0 <config file> <operations> <gpu number>"
    exit 1;
else 
    export CUDA_VISIBLE_DEVICES="$3";
fi


#PYTHONPATH=./:$PYTHONPATH python morphodecomp.py $1 $2
#PYTHONPATH=./:./test/:$PYTHONPATH python -m unittest test_morphochallenge.TestMorphoChallenge
PYTHONPATH=./:./test/:$PYTHONPATH python -m unittest test_morphochallenge.TestMorphoChallengeW2V
