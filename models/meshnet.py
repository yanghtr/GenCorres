import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from loguru import logger

from models.arap import ARAP
from models.asap import ASAP
from models.meshnet_base import MeshDecoder

from utils import geom_utils
from torch_cluster import knn
import pytorch3d.loss

class MeshNet(nn.Module):
    def __init__(self,
                 config,
                 dataset,
                 edge_index,
                 down_transform,
                 up_transform,
                ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.decoder = MeshDecoder(edge_index=edge_index,
                                   down_transform=down_transform,
                                   up_transform=up_transform,
                                   **config.model.mesh)

        self.use_mesh_arap_with_asap = config.loss.get('use_mesh_arap_with_asap', False)
        if self.use_mesh_arap_with_asap:
            self.arap = ASAP(dataset.template_faces, dataset.template_faces.max() + 1)
            logger.info("\nuse_mesh_arap_with_asap, weight_asap=0.05\n")
        else:
            self.arap = ARAP(dataset.template_faces, dataset.template_faces.max() + 1)

        self.data_mean_gpu = torch.from_numpy(dataset.mean_init).float()
        self.data_std_gpu = torch.from_numpy(dataset.std_init).float()

    def forward(self, lat_vecs, batch_dict, config, state_info=None):
        '''
        Args:
            lat_vecs: (B, latent_dim)
        '''
        mesh_out_pred = self.decoder(lat_vecs) # (B, N, 3), normalized coordinates
        batch_dict["mesh_verts_nml_pred"] = mesh_out_pred

        if state_info is not None:
            self.get_loss(lat_vecs, batch_dict, config, state_info)

        return batch_dict

    @staticmethod
    def get_point2plane_loss(pred_shape, gt_shape, gt_faces):
        '''
        Args:
            gt_shape: (B, Vx, 3)
            gt_faces: (B, F, 3)
            pred_shape: (B, Vy, 3)
        '''
        assert(gt_shape.shape[0] == pred_shape.shape[0])
        gt_fnormals = geom_utils.compute_face_normals(gt_shape, gt_faces)
        gt_vnormals = geom_utils.compute_vertex_normals(gt_shape, gt_faces, gt_fnormals)
        batch_size, num_x, num_y = gt_shape.shape[0], gt_shape.shape[1], pred_shape.shape[1]
        x = gt_shape.reshape(-1, 3) # (B*Vx, 3)
        y = pred_shape.reshape(-1, 3) # (B*Vy, 3)
        batch_x = torch.arange(batch_size)[:, None].expand(-1, num_x).reshape(-1).cuda() # (B*Vx)
        batch_y = torch.arange(batch_size)[:, None].expand(-1, num_y).reshape(-1).cuda() # (B*Vy)
        corres = knn(x, y, 1, batch_x, batch_y)
        diff = (x[corres[1]] - y[corres[0]]).reshape(batch_size, num_y, 3) # (B, Vy, 3)
        vnorm_corres = gt_vnormals.reshape(-1, 3)[corres[1]].reshape(batch_size, num_y, 3) # (B, Vy, 3)
        dist_plane = torch.abs(torch.sum(diff * vnorm_corres, dim=-1)) # (B, Vy)
        l1_loss = torch.mean(dist_plane)
        return l1_loss


    @staticmethod
    def get_jacobian_rand(cur_shape, z, data_mean_gpu, data_std_gpu, model, device, epsilon=[1e-3], nz_max=60):
        nb, nz = z.size()
        _, n_vert, nc = cur_shape.size()
        if nz >= nz_max:
          rand_idx = np.random.permutation(nz)[:nz_max]
          nz = nz_max
        else:
          rand_idx = np.arange(nz)
        
        jacobian = torch.zeros((nb, n_vert*nc, nz)).to(device)
        for i, idx in enumerate(rand_idx):
            dz = torch.zeros(z.size()).to(device)
            dz[:, idx] = epsilon
            z_new = z + dz
            out_new = model(z_new)
            shape_new = out_new * data_std_gpu + data_mean_gpu
            dout = (shape_new - cur_shape).view(nb, -1)
            jacobian[:, :, i] = dout/epsilon
        return jacobian


    def get_loss(self, lat_vecs, batch_dict, config, state_info):
        epoch = state_info['epoch']
        device = batch_dict['verts_init_nml'].device
        loss = torch.zeros(1, device=device) 
        self.data_std_gpu = self.data_std_gpu.to(device)
        self.data_mean_gpu = self.data_mean_gpu.to(device)
        assert(config.rep in ['mesh'])

        verts_pred = batch_dict['mesh_verts_nml_pred'] * self.data_std_gpu + self.data_mean_gpu # (B, V, 3)
        verts_init = batch_dict['verts_init_nml'] * self.data_std_gpu + self.data_mean_gpu
        verts_raw = batch_dict['verts_raw']
        faces_raw = batch_dict['faces_raw']
        assert(verts_raw.shape[0] == faces_raw.shape[0])
        verts_raw_lengths = batch_dict['verts_raw_lengths'] if 'verts_raw_lengths' in batch_dict else None

        # mesh init loss
        if config.use_point2point_loss:
            point2point_loss = F.l1_loss(verts_pred, verts_init, reduction='mean') * config.loss.point2point_loss_weight
            loss += point2point_loss
            batch_dict['point2point_loss'] = point2point_loss
            state_info['point2point_loss'] = point2point_loss.item()

        if config.use_point2plane_loss:
            raise NotImplementedError("Unbatched operation is not implemented")
            point2plane_loss = self.get_point2plane_loss(verts_pred, verts_raw, faces_raw) * config.loss.point2plane_loss_weight
            loss += point2plane_loss
            batch_dict['point2plane_loss'] = point2plane_loss
            state_info['point2plane_loss'] = point2plane_loss.item()

        if config.use_chamfer_loss:
            chamfer_loss, _ = pytorch3d.loss.chamfer_distance(x=verts_pred, y=verts_raw, x_lengths=None, y_lengths=verts_raw_lengths) * config.loss.chamfer_loss_weight
            loss += chamfer_loss
            batch_dict['chamfer_loss'] = chamfer_loss
            state_info['chamfer_loss'] = chamfer_loss.item()

        if config.use_mesh_arap:
            jacob = self.get_jacobian_rand(
                        verts_pred,  lat_vecs, self.data_mean_gpu, self.data_std_gpu,
                        self.decoder, device, epsilon=0.1, nz_max=config.loss.nz_max)
            try:
                arap_energy = self.arap(verts_pred, jacob, weight_asap=config.loss.mesh_weight_asap) / jacob.shape[-1]
            except:
                from IPython import embed; embed()
            mesh_arap_loss = arap_energy * config.loss.mesh_arap_weight

            loss += mesh_arap_loss
            batch_dict['mesh_arap_loss'] = mesh_arap_loss
            state_info['mesh_arap_loss'] = mesh_arap_loss.item()

        batch_dict["loss"] = loss




