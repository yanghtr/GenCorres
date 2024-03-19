function [] = main(start_idx, interval)
%% start_idx starts from 1
%% load data
addpath('io/');
meta_data_path = '../dfaust1k/meta_test_6499_K25.mat';
mesh_interp_dir = '../../work_dir/dfaust/ivae_dfaustJSM1k/results/test/interp_edges_sdf/6499/';
mesh_raw_dir = '../dfaust1k/mesh_raw/';
mesh_sim_dir = '../dfaust1k/mesh_sim/';
mesh_def_dir = get_directory('../dfaust1k/mesh_def/');
% log_dir = get_directory('../dfaust1k/log/');

meta_data = load(meta_data_path);
edge_ids = meta_data.edge_ids; % starts from 0
fids = meta_data.fids;
num_edges = size(edge_ids, 1);
num_meshes = size(fids, 1);
template_idx = meta_data.template_idx; % starts from 0
template_fid = meta_data.template_fid;
assert(max(max(edge_ids)) == num_meshes - 1);

template_raw_mesh = read_obj([mesh_raw_dir, template_fid, '.obj']);
template_sim_mesh = read_obj([mesh_sim_dir, template_fid, '_sim.obj']);

%% Hyperparameters
params = Params;
params.lambda = 10;
params.beta = 0.05;

%%
if isstring(start_idx) || ischar(start_idx)
    start_idx = str2num(start_idx)
end
if isstring(interval) || ischar(interval)
    interval = str2num(interval)
end
end_idx = min(start_idx + interval - 1, num_edges);
for eid = start_idx : end_idx
    % get pair id
    sid = edge_ids(eid, 1);
    tid = edge_ids(eid, 2);
    sfid = strtrim(fids(sid + 1, :)); % matlab starts from 1
    tfid = strtrim(fids(tid + 1, :)); % matlab starts from 1
    precheck_mesh_path = sprintf('./%s/meshdef_%d_%d.obj', mesh_def_dir, sid, tid);
    if isfile(precheck_mesh_path)
        continue;
    end
    mesh_src_sim = read_obj([mesh_sim_dir, sfid, '_sim.obj']);
    mesh_tgt_sim = read_obj([mesh_sim_dir, tfid, '_sim.obj']);
    mesh_src = read_obj([mesh_raw_dir, sfid, '.obj']);
    mesh_tgt = read_obj([mesh_raw_dir, tfid, '.obj']);
    % log file
    % log_path = sprintf('%s/%d.log', log_dir, eid);
    % fileID = fopen(log_path, 'w');
    % fprintf(fileID, '----- interp: eid=%d, sid=%d, tid=%d, sfid=%s, tfid=%s -----\n', eid, sid, tid, sfid, tfid);
    fprintf('----- interp: eid=%d, sid=%d, tid=%d, sfid=%s, tfid=%s -----\n', eid, sid, tid, sfid, tfid);
    % load interpolation meshes
    meshes_interp = cell(1, 9); 
    for i_interp = 1 : 9
        fpath = sprintf('%s/%d_%d/%d_%d_%02d.obj', mesh_interp_dir, sid, tid, sid, tid, i_interp);
        mesh_interp = read_obj(fpath);
        meshes_interp{i_interp} = mesh_interp;
    end
    % interpolate meshes
    mesh_src_sim_def = icp_interpolation(mesh_src_sim, mesh_tgt_sim, meshes_interp, params);
    % DEBUG: write_obj(mesh_src_sim_def, '/mnt/yanghaitao/Projects/GenCorres/gencorres/vis/registration/dfaust1k/mesh_def/mesh_src_sim_def.obj');
    % fprintf(fileID, 'ED + refine start: eid = %d, sid = %d, tid = %d\n', eid, sid, tid);
    fprintf('ED + refine start: eid = %d, sid = %d, tid = %d\n', eid, sid, tid);

    mesh_src_def = embedded_deformation(mesh_src, mesh_src_sim, mesh_src_sim_def);
    
    % DEBUG: write_obj(mesh_src_def, '/mnt/yanghaitao/Projects/GenCorres/gencorres/vis/registration/dfaust1k/mesh_def/ed.obj');

    poss_vec = non_rigid_icp2(mesh_src, mesh_tgt, mesh_src_def.vertexPoss, 1, 0.05, 10, 1);
    mesh_src_def.vertexPoss = poss_vec;

    dump_mesh_path = sprintf('./%s/meshdef_%d_%d.obj', mesh_def_dir, sid, tid); % starts from 0
    write_obj(mesh_src_def, dump_mesh_path);

    % fprintf(fileID, 'Done: eid = %d, sid = %d, tid = %d\n', eid, sid, tid);
    fprintf('Done: eid = %d, sid = %d, tid = %d\n', eid, sid, tid);
    % fclose(fileID);
end

end


