#!/bin/bash
export PATH=/scratch/cluster/yanght/Software/miniconda3/bin/:$PATH
source /scratch/cluster/yanght/Software/miniconda3/etc/profile.d/conda.sh
conda activate torch13;
which python
echo "setup finished!"

offset=$1
interval=100 # 10
start_idx=$((offset * interval))
echo "start_idx=${start_idx}, interval=${interval}"

python main.py --config ./config/dfaust/ivae_dfaustJSM1k.yaml --mode interp_edges --rep sdf --continue_from 6499 --split test --edge_ids_path ./work_dir/dfaust/ivae_dfaustJSM1k/results/test/analysis_sdf/edge_ids/test_6499_edge_ids_K25.npy --parallel_idx ${offset} --parallel_interval ${interval}
