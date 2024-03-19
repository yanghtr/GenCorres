function [] = write_obj(shape, filename)
%
f_id = fopen(filename, 'w');
for vId = 1 : size(shape.vertexPoss,2)
    pos = shape.vertexPoss(:, vId);
    fprintf(f_id, 'v %f %f %f\n', pos(1), pos(2), pos(3));
end
for fId = 1 : size(shape.faceVIds,2)
    vids = shape.faceVIds(:,fId);
    fprintf(f_id, 'f %d %d %d\n', vids(1), vids(2), vids(3));
end
fclose(f_id);