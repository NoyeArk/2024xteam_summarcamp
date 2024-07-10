#!/bin/bash
NUM_PROC=1
shift
F:/anaconda/anaconda3/envs/DeltaZero/python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --use_env train.py "$@"

