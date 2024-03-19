function [deformed_mesh] = embedded_deformation(mesh, mesh_sim, mesh_sim_deformed)
% Compute the deformation using mesh_sim and mesh_sim_deformed
% Use the resulting deformation to deform mesh to obtain deformed_mesh
[~, nIds] = compute_neighbors(mesh_sim);
cur_trans = vertex_trans_fitting(mesh_sim.vertexPoss, mesh_sim_deformed.vertexPoss, nIds);
[IDX, DIS] = knnsearch(mesh_sim.vertexPoss', mesh.vertexPoss', 'k', 20);
sigma = median(DIS(:,2));
Weights = exp(-(DIS.*DIS)/2/sigma/sigma);
sumOfWeights = sum(Weights')';
Weights = Weights./(sumOfWeights*ones(1,20));
%
deformed_mesh = mesh;
for id = 1 : size(mesh.vertexPoss, 2)
    tPos = zeros(3,1);
    sPos = mesh.vertexPoss(:, id);
    for i = 1 : size(IDX, 2)
        transId = IDX(id, i);
        w = Weights(id, i);
        A = cur_trans{transId}(:,1:3);
        b = cur_trans{transId}(:,4);
        tPos = tPos + w*(A*sPos + b);
    end
    deformed_mesh.vertexPoss(:, id) = tPos;
end

%%
function [A, nIds] = compute_neighbors(mesh)
numV = size(mesh.vertexPoss, 2);
edges = [mesh.faceVIds(1, :), mesh.faceVIds(2, :), mesh.faceVIds(3, :);
         mesh.faceVIds(2, :), mesh.faceVIds(3, :), mesh.faceVIds(1, :)];
A = sparse(edges(1, :), edges(2, :), ones(1, size(edges, 2)));
nIds = cell(1, numV);
for vId = 1 : numV
    nIds{vId} = find(A(vId,:));
end

%%
function [vertex_trans] = vertex_trans_fitting(fixed_poss, opt_poss, nIds)
%
numV = size(fixed_poss, 2);
for vId = 1 : numV
    ids = nIds{vId};
    valence = length(ids);
    P = double(fixed_poss(:,ids) - fixed_poss(:,vId)*ones(1,valence));
    Q = double(opt_poss(:,ids) - opt_poss(:,vId)*ones(1,valence));
    A = (Q*P')*pinv(P*P');
    b = opt_poss(:,vId) - A*fixed_poss(:,vId);
    vertex_trans{vId} = [A,b];
end
%
