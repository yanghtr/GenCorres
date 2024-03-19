import sys
import trimesh
import numpy as np
import scipy.io as sio
from loguru import logger

import torch
from torch_geometric.utils import degree, get_laplacian
import torch_sparse as ts


def get_laplacian_kron3x3(edge_index, edge_weights, N):
    edge_index, edge_weight = get_laplacian(edge_index, edge_weights, num_nodes=N) # (2, V+2E), (V+2E,)
    edge_weight *= 2
    e0, e1 = edge_index
    i0 = [e0*3, e0*3+1, e0*3+2]
    i1 = [e1*3, e1*3+1, e1*3+2]
    vals = [edge_weight, edge_weight, edge_weight]
    i0 = torch.cat(i0, 0)
    i1 = torch.cat(i1, 0)
    vals = torch.cat(vals, 0)
    indices, vals = ts.coalesce([i0, i1], vals, N*3, N*3) # (2, 3(V+2E)), (2(V+2E),)
    return indices, vals


def compute_asap3d_sparse(verts, faces, weight_asap=0.1):
    """ compute normalized: (L_arap + weight_asap * L_asap) / (1 + weight_asap)
    Args:
        verts: (N, 3)
        faces: (E, 3)
    Returns:
        Hessian: (3N, 3N), sparse
    """
    N = verts.shape[0]
    device = verts.device
    adj = torch.zeros((N, N), device=device)
    adj[faces[:, 0], faces[:, 1]] = 1
    adj[faces[:, 1], faces[:, 2]] = 1
    adj[faces[:, 0], faces[:, 2]] = 1
    adj = adj + adj.T
    edge_index = torch.as_tensor(torch.stack(torch.where(adj > 0), 0), dtype=torch.long) # (2, 2E)
    # edge_index = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0).T # (2, 2E)

    e0, e1 = edge_index # (2E,), (2E,)
    deg = degree(e0, N) # (V,)
    edge_weight = torch.ones_like(e0) # (2E,)
    edge_vecs = verts[e0, :] - verts[e1, :] # (2E, 3)
    edge_vecs_sq = (edge_vecs * edge_vecs).sum(-1) # (2E,)
    
    # XXXXXX COMPUTE L XXXXXX
    L_indices, L_vals = get_laplacian_kron3x3(edge_index, edge_weight, N) 
    L = torch.sparse_coo_tensor(L_indices, L_vals, (N*3, N*3))
    
    # XXXXXX COMPUTE B XXXXXX
    B0, B1, B_vals = [], [], []
    # off-diagonal: use e0 and e1
    B0.append(e0*3  ); B1.append(e1*3+1); B_vals.append(-edge_vecs[:, 2]*edge_weight)
    B0.append(e0*3  ); B1.append(e1*3+2); B_vals.append( edge_vecs[:, 1]*edge_weight)
    B0.append(e0*3+1); B1.append(e1*3+0); B_vals.append( edge_vecs[:, 2]*edge_weight)
    B0.append(e0*3+1); B1.append(e1*3+2); B_vals.append(-edge_vecs[:, 0]*edge_weight)
    B0.append(e0*3+2); B1.append(e1*3+0); B_vals.append(-edge_vecs[:, 1]*edge_weight)
    B0.append(e0*3+2); B1.append(e1*3+1); B_vals.append( edge_vecs[:, 0]*edge_weight)

    # in-diagonal: use e0 and e0
    B0.append(e0*3  ); B1.append(e0*3+1); B_vals.append(-edge_vecs[:, 2]*edge_weight)
    B0.append(e0*3  ); B1.append(e0*3+2); B_vals.append( edge_vecs[:, 1]*edge_weight)
    B0.append(e0*3+1); B1.append(e0*3+0); B_vals.append( edge_vecs[:, 2]*edge_weight)
    B0.append(e0*3+1); B1.append(e0*3+2); B_vals.append(-edge_vecs[:, 0]*edge_weight)
    B0.append(e0*3+2); B1.append(e0*3+0); B_vals.append(-edge_vecs[:, 1]*edge_weight)
    B0.append(e0*3+2); B1.append(e0*3+1); B_vals.append( edge_vecs[:, 0]*edge_weight)

    B0 = torch.cat(B0, 0); B1 = torch.cat(B1, 0); B_vals = torch.cat(B_vals, 0)
    B = torch.sparse_coo_tensor(torch.stack([B0, B1]), B_vals, (N*3, N*3))

    # XXXXXX COMPUTE H XXXXXX
    H0, H1, H_vals = [], [], []
    # i==j
    H0.append(e0*3  ); H1.append(e0); H_vals.append(-edge_vecs[:, 0]) 
    H0.append(e0*3+1); H1.append(e0); H_vals.append(-edge_vecs[:, 1]) 
    H0.append(e0*3+2); H1.append(e0); H_vals.append(-edge_vecs[:, 2]) 
    # (i, j) \in E
    H0.append(e0*3  ); H1.append(e1); H_vals.append(-edge_vecs[:, 0]) 
    H0.append(e0*3+1); H1.append(e1); H_vals.append(-edge_vecs[:, 1]) 
    H0.append(e0*3+2); H1.append(e1); H_vals.append(-edge_vecs[:, 2]) 

    H0 = torch.cat(H0, 0); H1 = torch.cat(H1, 0); H_vals = torch.cat(H_vals, 0)
    H = torch.sparse_coo_tensor(torch.stack([H0, H1]), H_vals, (N*3, N))
    
    # XXXXXX COMPUTE C XXXXXX
    C0, C1, C_vals = [], [], []
    for di in range(3):
        for dj in range(3):
            C0.append(e0*3+di); C1.append(e0*3+dj); C_vals.append(-edge_vecs[:, di]*edge_vecs[:, dj]*edge_weight)
        C0.append(e0*3+di); C1.append(e0*3+di); C_vals.append(edge_vecs_sq*edge_weight)
    C0 = torch.cat(C0, 0); C1 = torch.cat(C1, 0); C_vals = torch.cat(C_vals, 0)
    C_indices, C_vals = ts.coalesce([C0, C1], C_vals, N*3, N*3)
    Cinv_indices = C_indices
    try:
        Cinv_vals = C_vals.view(N, 3, 3).inverse().reshape(-1)
    except:
        logger.debug('Cinv_vals error: use pinv')
        Cinv_vals = torch.linalg.pinv(C_vals.view(N, 3, 3)).reshape(-1)
    Cinv = torch.sparse_coo_tensor(Cinv_indices, Cinv_vals, (N*3, N*3))

    # XXXXXX COMPUTE G XXXXXX
    G0, G1, G_vals = [], [], []
    # NOTE: DO NOT use: G_vals.append(1/(edge_vecs_sq*edge_weight)). Have to create the matrix first then inverse since 1/sum(v) != sum(1/v)
    G0.append(e0); G1.append(e0); G_vals.append(edge_vecs_sq*edge_weight)
    G0 = torch.cat(G0, 0); G1 = torch.cat(G1, 0); G_vals = torch.cat(G_vals, 0)
    G_indices, G_vals = ts.coalesce([G0, G1], G_vals, N, N)
    Ginv_indices = G_indices
    Ginv_vals = 1 / G_vals
    Ginv_vals[G_vals < 1e-6] = 0 # remove 1/0 = inf
    Ginv = torch.sparse_coo_tensor(Ginv_indices, Ginv_vals, (N, N))

    # XXXXXX COMPUTE Hessian XXXXXX
    BCinv = torch.sparse.mm(B, Cinv)
    BCinvBT = torch.sparse.mm(BCinv, B.t())

    HGinv = torch.sparse.mm(H, Ginv)
    HGinvHT = torch.sparse.mm(HGinv, H.t())

    Hessian_sparse = L - BCinvBT - weight_asap / (1 + weight_asap) * HGinvHT

    return Hessian_sparse


class ASAP(torch.nn.Module):
    def __init__(self, template_face, num_points):
        super(ASAP, self).__init__()
        N = num_points
        self.template_face = template_face # (F=13776, 3)
        adj = np.zeros((num_points, num_points))
        adj[template_face[:, 0], template_face[:, 1]] = 1
        adj[template_face[:, 1], template_face[:, 2]] = 1
        adj[template_face[:, 0], template_face[:, 2]] = 1
        adj = adj + adj.T
        edge_index = torch.as_tensor(np.stack(np.where(adj > 0), 0),
                                     dtype=torch.long) # (2, 2E=41328)
        e0, e1 = edge_index # (2E,), (2E,)
        deg = degree(e0, N) # (V,)
        edge_weight = torch.ones_like(e0) # (2E,)
        
        L_indices, L_vals = get_laplacian_kron3x3(edge_index, edge_weight, N) 
        self.register_buffer('L_indices', L_indices)
        self.register_buffer('L_vals', L_vals)
        self.register_buffer('edge_weight', edge_weight)
        self.register_buffer('edge_index', edge_index)
  
    def forward(self, x, J, k=0, weight_asap=0.05):
        """ compute normalized: (L_arap + weight_asap * L_asap) / (1 + weight_asap)
          x: [B, N, 3] point locations.
          J: [B, N*3, D] Jacobian of generator.
          J_eigvals: [B, D]
        """
        num_batches, N = x.shape[:2]
        e0, e1 = self.edge_index
        edge_vecs = x[:, e0, :] - x[:, e1, :] # (B, 2E, 3)
        trace_ = []
        
        for i in range(num_batches):
            LJ = ts.spmm(self.L_indices, self.L_vals, N*3, N*3, J[i]) # (3N, D)
            JTLJ = J[i].T.matmul(LJ)
  
            # XXXXXX COMPUTE B XXXXXX
            B0, B1, B_vals = [], [], []
            # off-diagonal: use e0 and e1
            B0.append(e0*3  ); B1.append(e1*3+1); B_vals.append(-edge_vecs[i, :, 2]*self.edge_weight)
            B0.append(e0*3  ); B1.append(e1*3+2); B_vals.append( edge_vecs[i, :, 1]*self.edge_weight)
            B0.append(e0*3+1); B1.append(e1*3+0); B_vals.append( edge_vecs[i, :, 2]*self.edge_weight)
            B0.append(e0*3+1); B1.append(e1*3+2); B_vals.append(-edge_vecs[i, :, 0]*self.edge_weight)
            B0.append(e0*3+2); B1.append(e1*3+0); B_vals.append(-edge_vecs[i, :, 1]*self.edge_weight)
            B0.append(e0*3+2); B1.append(e1*3+1); B_vals.append( edge_vecs[i, :, 0]*self.edge_weight)
  
            # in-diagonal: use e0 and e0
            B0.append(e0*3  ); B1.append(e0*3+1); B_vals.append(-edge_vecs[i, :, 2]*self.edge_weight)
            B0.append(e0*3  ); B1.append(e0*3+2); B_vals.append( edge_vecs[i, :, 1]*self.edge_weight)
            B0.append(e0*3+1); B1.append(e0*3+0); B_vals.append( edge_vecs[i, :, 2]*self.edge_weight)
            B0.append(e0*3+1); B1.append(e0*3+2); B_vals.append(-edge_vecs[i, :, 0]*self.edge_weight)
            B0.append(e0*3+2); B1.append(e0*3+0); B_vals.append(-edge_vecs[i, :, 1]*self.edge_weight)
            B0.append(e0*3+2); B1.append(e0*3+1); B_vals.append( edge_vecs[i, :, 0]*self.edge_weight)

            B0 = torch.cat(B0, 0)
            B1 = torch.cat(B1, 0)
            B_vals = torch.cat(B_vals, 0)
            B_indices, B_vals = ts.coalesce([B0, B1], B_vals, N*3, N*3)
            BT_indices, BT_vals = ts.transpose(B_indices, B_vals, N*3, N*3)

            # XXXXXX COMPUTE H XXXXXX
            H0, H1, H_vals = [], [], []
            # i==j
            H0.append(e0*3  ); H1.append(e0); H_vals.append(-edge_vecs[i, :, 0]*self.edge_weight) 
            H0.append(e0*3+1); H1.append(e0); H_vals.append(-edge_vecs[i, :, 1]*self.edge_weight) 
            H0.append(e0*3+2); H1.append(e0); H_vals.append(-edge_vecs[i, :, 2]*self.edge_weight) 
            # (i, j) \in E
            H0.append(e0*3  ); H1.append(e1); H_vals.append(-edge_vecs[i, :, 0]*self.edge_weight) 
            H0.append(e0*3+1); H1.append(e1); H_vals.append(-edge_vecs[i, :, 1]*self.edge_weight) 
            H0.append(e0*3+2); H1.append(e1); H_vals.append(-edge_vecs[i, :, 2]*self.edge_weight) 

            H0 = torch.cat(H0, 0); H1 = torch.cat(H1, 0); H_vals = torch.cat(H_vals, 0);
            H_indices, H_vals = ts.coalesce([H0, H1], H_vals, N*3, N)
            HT_indices, HT_vals = ts.transpose(H_indices, H_vals, N*3, N)
            
            # XXXXXX COMPUTE C XXXXXX
            C0, C1, C_vals = [], [], []
            edge_vecs_sq = (edge_vecs[i] * edge_vecs[i]).sum(-1)
            evi = edge_vecs[i] # (2E, 3)
            for di in range(3):
                for dj in range(3):
                    C0.append(e0*3+di); C1.append(e0*3+dj); C_vals.append(-evi[:, di]*evi[:, dj]*self.edge_weight)
                C0.append(e0*3+di); C1.append(e0*3+di); C_vals.append(edge_vecs_sq*self.edge_weight)
            C0 = torch.cat(C0, 0); C1 = torch.cat(C1, 0); C_vals = torch.cat(C_vals, 0)
            C_indices, C_vals = ts.coalesce([C0, C1], C_vals, N*3, N*3)
            Cinv_indices = C_indices
            try:
                Cinv_vals = C_vals.view(N, 3, 3).inverse().reshape(-1)
            except:
                logger.debug('C_vals error: use pinv')
                Cinv_vals = torch.linalg.pinv(C_vals.view(N, 3, 3)).reshape(-1)

            # XXXXXX COMPUTE G XXXXXX
            G0, G1, G_vals = [], [], []
            # NOTE: DO NOT use: G_vals.append(1/(edge_vecs_sq*edge_weight)). Have to create the matrix first then inverse since 1/sum(v) != sum(1/v)
            G0.append(e0); G1.append(e0); G_vals.append(edge_vecs_sq*self.edge_weight)
            G0 = torch.cat(G0, 0); G1 = torch.cat(G1, 0); G_vals = torch.cat(G_vals, 0)
            G_indices, G_vals = ts.coalesce([G0, G1], G_vals, N, N)
            Ginv_indices = G_indices
            Ginv_vals = 1 / G_vals
            Ginv_vals[G_vals < 1e-6] = 0 # remove 1/0 = inf

            # XXXXXX COMPUTE Hessian XXXXXX
            BTJ = ts.spmm(BT_indices, BT_vals, N*3, N*3, J[i])
            CinvBTJ = ts.spmm(Cinv_indices, Cinv_vals, N*3, N*3, BTJ)
            JTBCinvBTJ = BTJ.T.mm(CinvBTJ)
 
            HTJ = ts.spmm(HT_indices, HT_vals, N, N*3, J[i])
            GinvHTJ = ts.spmm(Ginv_indices, Ginv_vals, N, N, HTJ)
            JTHGinvHTJ = HTJ.T.mm(GinvHTJ)

            Rm = JTLJ - JTBCinvBTJ - weight_asap / (1 + weight_asap) * JTHGinvHTJ 

            e = torch.linalg.eigvalsh(Rm).clip(0)
            e = e ** 0.5

            trace = e.sum()
            trace_.append(trace)

        trace_ = torch.stack(trace_, )
        return trace_.mean()


if __name__ == '__main__':

    mesh = trimesh.load('./mesh.obj', process=False)
    # mesh = trimesh.load('./meshsim.obj', process=False)
    template_faces = np.asarray(mesh.faces)
    x = np.asarray(mesh.vertices)
    N = x.shape[0]

    x = torch.from_numpy(x)[None, ...]
    J = torch.randn(1, N*3, 16)
    arap = ASAP(template_faces, N)
    arap(x, J)





