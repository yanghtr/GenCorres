function compute_GDM(start_idx, interval)

addpath('io/');
addpath("geodesic/");
meta_data_path = '../dfaust1k/meta_test_6499_K5.mat';
mesh_raw_dir = '../dfaust1k/mesh_raw/';
dist_mat_dir = get_directory('/media/yanghaitao/HaitaoYang/Graphicsai_Backup/mnt/yanghaitao/Dataset/DFAUST/dfaust1k/distance_matrix');

meta_data = load(meta_data_path);
fids = meta_data.fids;
num_meshes = size(fids, 1);
template_idx = meta_data.template_idx; % starts from 0
template_fid = meta_data.template_fid;

if isstring(start_idx) || ischar(start_idx)
        start_idx = str2num(start_idx)                
end
if isstring(interval) || ischar(interval)
        interval = str2num(interval)
end
end_idx = min(start_idx + interval - 1, num_meshes);
for idx = start_idx : end_idx
    fprintf("start compute_dist_matrix: %d", idx);
    fid = strtrim(fids(idx, :)); % matlab starts from 1
    
    [X.vert, X.triv] = read_obj_nm([mesh_raw_dir, '/', fid, '.obj']); % NeuroMorph API to load obj
    X.vert = X.vert';
    X.triv = X.triv';
    X.n = size(X.vert, 1);
    X.m = size(X.triv, 1);

    D = compute_dist_matrix(X);
    D = single(D);

    save([dist_mat_dir, '/', num2str(idx - 1), '.mat'], 'D');
    fprintf("finish compute_dist_matrix: %d (matlab), %d (python)\n", idx, idx-1);
end

end
