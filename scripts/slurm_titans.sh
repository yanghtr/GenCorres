#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
PY_ARGS=${@:4}

CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:${GPUS} \
    --mem-per-cpu=6G \
    --cpus-per-task=$((CPUS_PER_TASK * GPUS)) \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ${PY_ARGS}
