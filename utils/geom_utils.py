#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F

import trimesh
import numpy as np
import networkx as nx
import point_cloud_utils as pcu

def index(x, idxs, dim):
    ''' Index a tensor along a given dimension using an index tensor, replacing
    the shape along the given dimension with the shape of the index tensor.
    Example:
        x:    [8, 6890, 3]
        idxs: [13776, 3]
        y = index(x, idxs, dim=1) -> y: [B, 13776, 3, 3]
        with each y[b, i, j, k] = x[b, idxs[i, j], k]
    '''
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*idxs.shape]
    return x.index_select(dim, idxs.view(-1)).reshape(target_shape)


def compute_face_normals(v, vi):
    '''
    @Args:
        v: (B, V, 3)
        vi: (B, F, 3)
    @Returns:
        face_normals: (B, F, 3)
    '''
    B = v.shape[0]
    vi = vi.expand(B, -1, -1)
    
    # p0 = torch.stack([index(v[i], vi[i, :, 0], 0) for i in range(b)])
    # p1 = torch.stack([index(v[i], vi[i, :, 1], 0) for i in range(b)])
    # p2 = torch.stack([index(v[i], vi[i, :, 2], 0) for i in range(b)])
    p0 = torch.stack([v[i].index_select(0, vi[i, :, 0]) for i in range(B)])
    p1 = torch.stack([v[i].index_select(0, vi[i, :, 1]) for i in range(B)])
    p2 = torch.stack([v[i].index_select(0, vi[i, :, 2]) for i in range(B)])
    v0 = p1 - p0
    v1 = p2 - p0
    n = torch.cross(v0, v1, dim=-1)
    return F.normalize(n, dim=-1)


def compute_vertex_normals(v, vi, fn):
    '''
    @Args:
        v: (B, V, 3), vertex coordinates
        vi: (B, F, 3), vertex indices
        fn: (B, F, 3), face normals
    @Returns:
        vn: (B, V, 3), vertex normals
    '''
    fn_exp = fn[:, :, None, :].expand(-1, -1, 3, -1).reshape(fn.shape[0], -1, 3) # repeat 3 times for 3 vertices of a face
    vi_flat = vi.view(vi.shape[0], -1).expand(v.shape[0], -1)
    vn = torch.zeros_like(v)

    for j in range(3):
        vn[..., j].scatter_add_(1, vi_flat, fn_exp[..., j])
    norm = torch.norm(vn, dim=-1, keepdim=True)
    vn = vn / norm.clamp(min=1e-8)
    return vn


def subdivide_mesh(mesh):
    '''
    Args:
        mesh: open3d.geometry.TriangleMesh
    '''
    mesh_sub = mesh.subdivide_midpoint(number_of_iterations=2)
    return mesh_sub


def sample_mesh(mesh):
    '''
    Args:
        mesh: open3d.geometry.TriangleMesh
        pcd:  open3d.geometry.PointCloud
    '''
    pcd = mesh.sample_points_uniformly(2000)
    return pcd


###################### Geometry ######################

def get_neighs_1ring(mesh):
    '''
    Args:
        mesh: trimesh.Trimesh
    Returns:
        neighs_1ring/(nIds): list of list.
    '''
    g = nx.from_edgelist(mesh.edges_unique)
    neighs_1ring = [list(g[i].keys()) for i in range(len(mesh.vertices))]
    return neighs_1ring


def vertex_trans_fitting(fixed_poss, opt_poss, nIds):
    '''
    Args:
        fixed_poss: (n, 3)
        opt_poss: (n, 3)
        nIds: neighs_1ring, list of list
    '''
    assert(fixed_poss.shape == opt_poss.shape)
    numV = fixed_poss.shape[0]

    vertex_trans = []
    for vId in range(numV):
        ids = nIds[vId]
        valence = len(ids)

        P = fixed_poss[ids, :] - fixed_poss[vId] # (v, 3)
        Q = opt_poss[ids, :] - opt_poss[vId] # (v, 3)
        A = (Q.T @ P) @ np.linalg.pinv(P.T @ P) # (3, 3)
        b = opt_poss[vId].reshape(-1, 1) - A @ fixed_poss[vId].reshape(-1, 1) # (3, 1)
        vertex_trans.append( np.concatenate((A, b), axis=-1) ) # (3, 4)

    return vertex_trans


def embedded_deformation(mesh, mesh_sim, mesh_sim_def, k=20):
    '''
    Args:
        mesh: trimesh, (n, 3)
        mesh_sim: trimesh, (ns, 3)
        mesh_sim_def: trimesh, (ns, 3)
    Returns:
        mesh_def: trimesh, (n, 3)
    '''

    nIds = get_neighs_1ring(mesh_sim)

    cur_trans = vertex_trans_fitting(mesh_sim.vertices, mesh_sim_def.vertices, nIds)

    DIS, IDX = pcu.k_nearest_neighbors(mesh.vertices, mesh_sim.vertices, k=k) # dense to sparse, (n, k), (n, k)

    sigma = np.median(DIS[:, 1]) # median of all NN distance. NOTE: 1 is better than 0
    weights = np.exp(-(DIS * DIS / 2 / sigma / sigma)) # (n, k)
    weights = weights / np.sum(weights, axis=-1, keepdims=True) # (n, k)

    mesh_def_verts = np.zeros_like(mesh.vertices) # (n, 3)
    for iv in range(mesh.vertices.shape[0]):
        tPos = np.zeros((3, 1))
        sPos = mesh.vertices[iv].reshape(3, 1) # (3, 1)
        for ik in range(k):
            transId = IDX[iv, ik]
            w = weights[iv, ik]
            A = cur_trans[transId][:, 0:3] # (3, 3)
            b = cur_trans[transId][:, 3:4] # (3, 1)
            tPos = tPos + w * (A @ sPos + b) # (3, 1)
        mesh_def_verts[iv] = tPos.reshape(-1)

    mesh_def = trimesh.Trimesh(vertices=mesh_def_verts, faces=mesh.faces, process=False)
    return mesh_def



if __name__ == '__main__':
    # import io_utils
    # v, vt, vi, vti = io_utils.read_obj_uv('../smpl_uv.obj')
    # fn = compute_face_normals(torch.FloatTensor(v[None, ...]), torch.LongTensor(vi[None, ...]))

    from IPython import embed; embed()

