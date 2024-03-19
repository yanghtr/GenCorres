#!/bin/bash
offset=$1
interval=10 # 6213
start_idx=$((offset * interval + 1))
echo "start_idx=${start_idx}, interval=${interval}"
/lusr/share/software/matlab-r2018b/bin/matlab -nodesktop -nosplash -r "main ${start_idx} ${interval}"
