%%
function [mesh_src_def] = icp_interpolation(mesh_src, mesh_tgt, meshes_interp, params)
    num_interp = length(meshes_interp);
    poss_vec = mesh_src.vertexPoss;
    % nonrigid registration to interpolation
    for i_interp = 1 : num_interp
        fprintf('i_interp = %d\n', i_interp);
        lambda_w = params.lambda;
        beta = params.beta;
        outerIterMax = 8;
        innerIterMax = 1;
        poss_vec = non_rigid_icp2(mesh_src, meshes_interp{i_interp}, poss_vec, lambda_w, beta, outerIterMax, innerIterMax);
        % DEBUG
        % mesh_tmp = mesh_src; mesh_tmp.vertexPoss = poss_vec;
        % write_obj(mesh_tmp, ['/mnt/yanghaitao/Projects/GenCorres/gencorres/vis/registration/dfaust1k/mesh_def/tmp', num2str(i_interp), '.obj']);       
    end
    % nonrigid registration to target
    lambda_w = params.lambda * 0.1;
    beta = params.beta;
    outerIterMax = 10;
    innerIterMax = 1;
    poss_vec = non_rigid_icp2(mesh_src, mesh_tgt, poss_vec, lambda_w, beta, outerIterMax, innerIterMax);

    mesh_src_def = mesh_src;    
    mesh_src_def.vertexPoss = poss_vec;   
end

