function [optimized_poss] = non_rigid_icp2(template, target, init_poss_vec, lambda, beta, outerIterMax, innerIterMax)
% Optimize the alignment between the template model and the target model
% Using the initial pose_vec
cur_poss = init_poss_vec;
numV = size(init_poss_vec, 2);
%
ids1 = template.faceVIds(1,:);
ids2 = template.faceVIds(2,:);
ids3 = template.faceVIds(3,:);
ROWs = [ids1;ids2;ids3];
COLs = [ids2;ids3;ids1];
VALs = ones(3,1)*ones(1,length(ids1));
A = sparse(ROWs, COLs, VALs, numV,numV);
A = max(A, A');
[i,j,~] = find(A);
edges = [i';j'];
for vId = 1 : numV
    nIds{vId} = find(A(vId,:));
end
cur_rotations = vertex_rotation_fitting(template.vertexPoss, cur_poss,...
    nIds);
%
target.vertexNors = compute_vertex_normal(target);
for outerIter = 1 : outerIterMax % 16
    % Compute bidirectional correspondences
    corres = bidirectional_corres(cur_poss, target.vertexPoss);
    fprintf(' outerIter = %d\n', outerIter);
    for innerIter = 1 : innerIterMax
        % Perform Gauss-Newton optimization to solve the induced
        % optimization problem
        [cur_poss, cur_rotations, flag] = one_step_non_rigid_icp(...
            corres, target.vertexPoss, target.vertexNors, edges, template.vertexPoss, cur_poss,...
            cur_rotations, lambda, beta);
        if flag == 0
            break;
        end
    end
end
optimized_poss = cur_poss;

% Perform one-step non-rigid ICP
function [next_poss, next_rotations, flag] = one_step_non_rigid_icp(...
            corres, target_poss, target_nors, edges, ori_poss, cur_poss,...
            cur_rotations, lambda, beta)
        %
%
[H_data, g_data] = data_term(corres, target_poss, target_nors, cur_poss, beta);
[H_def, g_def] = deformation_term(edges, ori_poss, cur_poss,...
    cur_rotations);
H = H_data + H_def*lambda;
g = g_data + g_def*lambda;
%
e_cur = energy(corres, target_poss, target_nors, edges, ori_poss, cur_poss,...
    cur_rotations, lambda, beta);
dx = H\g;
if norm(dx) < 1e-6
    flag = 0;
    next_poss = cur_poss;
    next_rotations = cur_rotations;
    return;
end
[next_poss, next_rotations] = update_variables(cur_poss,...
    cur_rotations, dx);
e_next = energy(corres, target_poss, target_nors, edges, ori_poss, next_poss,...
    next_rotations, lambda, beta);
if e_next < e_cur
    fprintf('  e_cur = %f, e_next = %f.\n', e_cur, e_next);
    flag = 1;
else
    flag = 0;
    s = mean(diag(H))*1e-6;
    dim2 = size(H,2);
    for iter = 1 : 12
        dx = (H+s*sparse(1:dim2,1:dim2,ones(1,dim2)))\g;
        [next_poss, next_rotations] = update_variables(cur_poss,...
            cur_rotations, dx);
        e_next = energy(corres, target_poss, target_nors, edges, ori_poss, next_poss,...
            next_rotations, lambda, beta);
        if e_next < e_cur
            flag = 1;
            fprintf('  e_cur = %f, e_next = %f.\n', e_cur, e_next);
            break;
        end
        s = s*4;
    end
end

% Update the solution based on the current solution
function [next_poss, next_rotations] = update_variables(cur_poss,...
    cur_rotations, dx)
%
numV = size(cur_poss, 2);
next_poss = cur_poss + reshape(dx(1:(3*numV)), [3,numV]);
for vId = 1 : numV
    rowIds = 3*numV + ((3*vId-2):(3*vId));
    dc = dx(rowIds);
    dR = expm([0 -dc(3) dc(2);
        dc(3) 0 -dc(1);
        -dc(2) dc(1) 0]);
    next_rotations{vId} = dR*cur_rotations{vId};
end

% The data-term
function [H_data, g_data] = data_term(corres, target_poss, target_nors, cur_poss, beta)
%
numV = size(cur_poss, 2);
dim = 3*numV;
numCorres = size(corres, 2);
t = double(reshape(target_poss(:, corres(2,:)) - cur_poss(:,corres(1,:)),...
    [3*numCorres,1]));
tp = 3*kron(corres(1,:), ones(1,3))...
    + kron(ones(1,numCorres),[-2,-1,0]);
J = sparse(1:(3*numCorres), tp, ones(1,3*numCorres), 3*numCorres, 2*dim); % 2*dim: (delta_p, c) \in R^6*numV

% point-2-point term
H_data = J'*J;
g_data = J'*t;
% point-2-plane term
plane_dis = double(sum((target_poss(:, corres(2,:)) - cur_poss(:,corres(1,:))).*target_nors(:,corres(2,:))));
J2 = sparse((1:numCorres)'*ones(1,3),...
    3*kron(corres(1,:)',ones(1,3)) + kron(ones(numCorres,1), [-2,-1,0]),...
    target_nors(:,corres(2,:))', numCorres, 2*dim);
H_data = H_data*beta + (J2'*J2)*(1-beta);
g_data = g_data*beta + (J2'*plane_dis')*(1-beta);

% The deformation term
function [H_def, g_def] = deformation_term(edges, ori_poss, cur_poss,...
    cur_rotations)
%
numV = size(ori_poss,2);
numE = size(edges, 2);
rowsJ = (1:(3*numE))'*ones(1,5);
colsJ = zeros(3*numE, 5);
valsJ = zeros(3*numE, 5);
% dxi - dxj + [] x c - (R_i^c(pi0-pj0) - (pic-pjc))
colsJ(:,1) = 3*kron(edges(1,:)', ones(3,1))+kron(ones(numE,1), [-2,-1,0]');
valsJ(:,1) = ones(3*numE,1);
colsJ(:,2) = 3*kron(edges(2,:)', ones(3,1))+kron(ones(numE,1), [-2,-1,0]');
valsJ(:,2) = -ones(3*numE,1);
colsJ(:,3:5) = 3*numV + 3*kron(edges(1,:)', ones(3,3))...
   + kron(ones(numE,1), ones(3,1)*[-2,-1,0]);
vec_d = zeros(3*numE,1);
for eId = 1 : numE
    rowIds = (3*eId-2):(3*eId);
    sId = edges(1, eId);
    tId = edges(2, eId);
    vec_o_trans = cur_rotations{sId}*(ori_poss(:,sId) - ori_poss(:,tId));
    vec_cur = (cur_poss(:,sId) - cur_poss(:,tId));
    vec_d(rowIds) = vec_o_trans - vec_cur;
    valsJ(rowIds,3:5) = [0 -vec_o_trans(3) vec_o_trans(2);
        vec_o_trans(3) 0 -vec_o_trans(1);
        -vec_o_trans(2) vec_o_trans(1) 0];
end
J = sparse(rowsJ, colsJ, valsJ, 3*numE, 6*numV);
H_def = J'*J;
g_def = J'*vec_d;

% Compute the energy for non-rigid registration
function [e] = energy(corres, target_poss, target_nors, edges, ori_poss, cur_poss, cur_rotations, lambda, beta)
% Compute the cumulative squared distance
dif = cur_poss(:, corres(1,:)) - target_poss(:, corres(2,:));
dis_plane = sum(dif.*target_nors(:,corres(2,:)));
e = sum(sum(dif.*dif))*beta + sum(dis_plane.*dis_plane)*(1-beta);
% Compute the rotation fitting residuals
for eId = 1 : size(edges,2)
    sId = edges(1, eId);
    tId = edges(2, eId);
    P = ori_poss(:,sId) - ori_poss(:,tId);
    Q = cur_poss(:,sId) - cur_poss(:,tId);
    dif = cur_rotations{sId}*P - Q;
    e = e + lambda*(dif'*dif);
end


% Compute bi-directional correspondences
function [corres] = bidirectional_corres(opt_poss, target_poss)
%
IDX1 = knnsearch(target_poss', opt_poss');
IDX2 = knnsearch(opt_poss', target_poss');
corres = [1:length(IDX1),IDX2';IDX1',1:length(IDX2)];

%
function [vertex_rotations] = vertex_rotation_fitting(fixed_poss, opt_poss, nIds)
% Use ARAP as initialization for R instead of using Identity
numV = size(fixed_poss, 2);
for vId = 1 : numV
    ids = nIds{vId};
    valence = length(ids);
    P = double(fixed_poss(:,ids) - fixed_poss(:,vId)*ones(1,valence));
    Q = double(opt_poss(:,ids) - opt_poss(:,vId)*ones(1,valence));
    S = P*Q';
    [u,v,w] = svd(S);
    R = w*u';
    if det(S) < 0
        R = w*diag([1,1,-1])*u';
    end
    vertex_rotations{vId} = R;
end
%
function [vertex_normal] = compute_vertex_normal(mesh)
%
p1 = mesh.vertexPoss(:,mesh.faceVIds(1,:));
p2 = mesh.vertexPoss(:,mesh.faceVIds(2,:));
p3 = mesh.vertexPoss(:,mesh.faceVIds(3,:));
e12 = p1 - p2;
e13 = p1 - p3;
facenors = cross(e12, e13);
lens = sqrt(sum(facenors.*facenors)) + 1e-10;
facenors = facenors./(ones(3,1)*lens);
numV = size(mesh.vertexPoss, 2);
vertex_normal = zeros(3, numV);
for fId = 1 : size(facenors,2)
    ids = mesh.faceVIds(:, fId);
    vertex_normal(:,ids(1)) = vertex_normal(:,ids(1)) + facenors(:,fId);
    vertex_normal(:,ids(2)) = vertex_normal(:,ids(2)) + facenors(:,fId);
    vertex_normal(:,ids(3)) = vertex_normal(:,ids(3)) + facenors(:,fId);
end
lens = sqrt(sum(vertex_normal.*vertex_normal)) + 1e-10;
vertex_normal = vertex_normal./(ones(3,1)*lens);
