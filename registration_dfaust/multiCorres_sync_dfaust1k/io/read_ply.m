function [mesh] = read_ply(filename)
%
temp = plyread(filename);
numV = length(temp.vertex.x);
numF = length(temp.face.vertex_indices);
%
mesh.vertexPoss = zeros(3, numV);
mesh.faceVIds = zeros(3, numF);
%
for id = 1 : numV
    mesh.vertexPoss(1, id) = temp.vertex.x(id);
    mesh.vertexPoss(2, id) = temp.vertex.y(id);
    mesh.vertexPoss(3, id) = temp.vertex.z(id);
end
for id = 1 : numF
    mesh.faceVIds(:, id) = temp.face.vertex_indices{id}'+1;
end
%
