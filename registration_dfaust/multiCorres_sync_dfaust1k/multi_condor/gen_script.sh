num_edges=31189
num_machine=10
interval=10
num_queue=$((num_edges/num_machine/interval + 1))
num_jobs_per_mac=$((num_queue * interval))
for i in $(seq 1 ${num_machine}); do
echo ${i};
mkdir log_${i};
echo -e "#!/bin/bash
start=\$1
offset=$((1 + (i-1) * num_jobs_per_mac))
start_idx=\$((start * ${interval} + offset))
echo \"start_idx=\${start_idx}, interval=${interval}\"
/lusr/share/software/matlab-r2018b/bin/matlab -c /lusr/share/software/matlab-r2018a/licenses/network.lic -nodesktop -nosplash -r \"main \${start_idx} ${interval}\" " >> batch_icp_${i}.sh;

echo -e "+Group = \"GRAD\"
+Project = \"GRAPHICS_VISUALIZATION\"
+ProjectDescription = \"corres\"
Universe     = vanilla
requirements = InMastodon
Executable   = ./multi_condor/batch_icp_${i}.sh
Output       = ./multi_condor/log_${i}/\$(Process).out
Error        = ./multi_condor/log_${i}/\$(Process).err
Log          = ./multi_condor/log_${i}/\$(Process).log
arguments = \$(Process)
Queue ${num_queue}" >> condor_${i}.sh;
chmod 777 batch_icp_${i}.sh;
chmod 777 condor_${i}.sh;
done
