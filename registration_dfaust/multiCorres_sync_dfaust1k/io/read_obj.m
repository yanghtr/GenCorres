function [Shape] = read_obj(filename)
%
numV = 100000;
numF = 200000;
vPos = zeros(3, numV);
vFace = zeros(3, numF);
vId = 0;
fId = 0;
f_id = fopen(filename, 'r');
while 1
    tline = fgetl(f_id);
    if tline == -1
        break;
    end
    if length(tline) < 2
        continue;
    end
    if tline(1) == 'v' && tline(2) == ' '
        vId = vId + 1;
        p = str2num(tline(3:length(tline)));
        vPos(:, vId) = p';
    end
    if tline(1) == 'f' && tline(2) == ' '
        fId = fId + 1;
        v = str2num(tline(3:length(tline)));
        vFace(:, fId) = v';
    end
end
fclose(f_id);
%
vPos = vPos(:,1:vId);
vFace = vFace(:,1:fId);
v1 = vFace(1,:);
v2 = vFace(2,:);
v3 = vFace(3,:);
edges = [v1,v2,v3;v2,v3,v1];
G = sparse(edges(1,:), edges(2,:), ones(1,size(edges,2)), vId, vId);
G = max(G,G');
[rows, cols, vals] = find(G);
edges = [rows';cols'];
Shape.vertexPoss = vPos;
Shape.faceVIds = vFace;
Shape.edges = edges;