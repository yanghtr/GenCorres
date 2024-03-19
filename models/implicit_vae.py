import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh
import numpy as np

from models.saldnet import SimplePointnet, ImplicitMap
from utils.diff_operators import gradient
from utils import implicit_utils
from utils.time_utils import *
from models.asap import compute_asap3d_sparse

import torch_sparse as ts


class ImplicitGenerator(nn.Module):
    def __init__(self,
                 config,
                 dataset,
                 **kwargs,
                ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.model_cfg = config.model.sdf
        self.auto_decoder = self.model_cfg.auto_decoder
        self.latent_size = self.model_cfg.decoder.latent_size
        self.with_normals = self.model_cfg.encoder.with_normals

        encoder_input_size = 6 if self.with_normals else 3

        self.encoder = SimplePointnet(dim=encoder_input_size, hidden_dim=2 * self.latent_size, c_dim=self.latent_size) if not self.auto_decoder else None

        self.decoder = ImplicitMap(**self.model_cfg.decoder)


    def forward(self, latent, batch_dict, config, state_info=None, only_encoder_forward=False, only_decoder_forward=False):
        '''
        Args:
            latent: (B, latent_dim), in this VAE model, should be the precomputed latent from forward only
        Notes:
            latent is generated from encoder or nn.Parameter that can be initialized
        '''
        points_mnfld = batch_dict['points_mnfld'] # (B, S, 3)
        normals_mnfld = batch_dict['normals_mnfld'] # (B, S, 3)
        points_nonmnfld = batch_dict['samples_nonmnfld'][:, :, :3].clone().detach().requires_grad_(True) # (B, S, 3)

        # get latent
        if self.encoder is not None and not only_decoder_forward: # VAE model
            assert(latent is None)
            encoder_input = torch.cat([points_mnfld, normals_mnfld], dim=-1) if self.with_normals else points_mnfld
            q_latent_mean, q_latent_std = self.encoder(encoder_input)

            q_z = torch.distributions.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = (q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))

            batch_dict['latent'] = latent # (B, latent_dim)
            batch_dict['latent_reg'] = latent_reg # (B,)
            batch_dict['q_latent_mean'] = q_latent_mean # (B, latent_dim)

            if only_encoder_forward:
                # IMPORTANT: use q_latent_mean instead of latent
                # return latent, q_latent_mean, torch.exp(q_latent_std)
                return q_latent_mean, q_latent_mean, torch.exp(q_latent_std)
        else:
            if only_encoder_forward:
                return None, None, None

        # decode latent to sdf
        assert(latent is not None)
        if only_decoder_forward:
            # NOTE: check using points_nonmnfld or points_mnfld
            sdf_nonmnfld = self.decoder(points_nonmnfld, latent)
            batch_dict['latent_norm_sq'] = (latent ** 2).mean(-1)
        else:
            sdf_nonmnfld = self.decoder(points_nonmnfld, latent)
        batch_dict['points_nonmnfld'] = points_nonmnfld # (B, S, 3)
        batch_dict['sdf_nonmnfld'] = sdf_nonmnfld # (B, S, 1)

        if state_info is not None:
            self.get_loss(latent, batch_dict, config, state_info)

        if config.mode == 'analysis':
            self.analysis(latent, batch_dict, config, state_info)

        return batch_dict


    def get_loss(self, latent, batch_dict, config, state_info):
        epoch = state_info['epoch']
        device = batch_dict['points_mnfld'].device
        loss = torch.zeros(1, device=device) 
        assert(config.rep in ['sdf'])

        # sdf loss
        sdf_loss_type = config.loss.get('sdf_loss_type', 'L1')
        if sdf_loss_type == 'L1':
            sdf_loss = F.l1_loss(batch_dict['sdf_nonmnfld'][:, :, 0].abs(), batch_dict['samples_nonmnfld'][:, :, -1])
            sdf_loss = sdf_loss * config.loss.sdf_weight
        else:
            raise NotImplementedError

        loss += sdf_loss
        batch_dict['sdf_loss'] = sdf_loss
        state_info['sdf_loss'] = sdf_loss.item()

        # VAE latent reg loss
        if config.use_sdf_latent_reg and 'latent_reg' in batch_dict and config.loss.vae_latent_reg_weight > 0: # True when VAE
            vae_latent_reg_loss = batch_dict['latent_reg'].mean() * config.loss.vae_latent_reg_weight
            loss += vae_latent_reg_loss
            batch_dict['vae_latent_reg_loss'] = vae_latent_reg_loss
            state_info['vae_latent_reg_loss'] = vae_latent_reg_loss.item()

        # AD latent reg loss
        if config.use_sdf_latent_reg and 'latent_norm_sq' in batch_dict and config.loss.ad_latent_reg_weight > 0: # True when AD
            ad_latent_reg_loss = batch_dict['latent_norm_sq'].mean() * config.loss.ad_latent_reg_weight # * min(1, epoch / 100)
            loss += ad_latent_reg_loss
            batch_dict['ad_latent_reg_loss'] = ad_latent_reg_loss
            state_info['ad_latent_reg_loss'] = ad_latent_reg_loss.item()

        if config.use_sdf_grad and config.loss.grad_loss_weight > 0:
            grad_nonmnfld = gradient(batch_dict['sdf_nonmnfld'], batch_dict['points_nonmnfld']) # (B, S, 3)
            normals_nonmnfld_gt = batch_dict['samples_nonmnfld'][:, :, 3:6] # (B, S, 3)

            grad_loss = torch.min(torch.abs(grad_nonmnfld - normals_nonmnfld_gt).sum(-1),
                                  torch.abs(grad_nonmnfld + normals_nonmnfld_gt).sum(-1)).mean()
            grad_loss = grad_loss * config.loss.grad_loss_weight
            loss += grad_loss
            batch_dict['grad_loss'] = grad_loss
            state_info['grad_loss'] = grad_loss.item()

        # sdf asap loss
        if config.use_sdf_asap:

            q_latent_mean = batch_dict['q_latent_mean'] # (B, latent_dim)
            assert(q_latent_mean.shape == latent.shape)
            assert(len(latent.shape) == 2)
            B = latent.shape[0]

            # sample latents
            sample_latent_space = config.loss.get('sample_latent_space', None)
            assert(sample_latent_space is not None)
            if sample_latent_space:

                sample_latent_space_type = config.loss.get('sample_latent_space_type', 'line')
                sample_latent_space_detach = config.loss.get('sample_latent_space_detach', False)
                if sample_latent_space_type == 'line':
                    line_range = config.loss.get('line_range', 1.0)
                    extra_half = (line_range - 1.0) / 2.0
                    rand_idx = np.random.choice(B, size=(B,))
                    rand_ratio = torch.rand((B, 1), device=device)
                    rand_ratio = rand_ratio * line_range - extra_half
                    batch_vecs = latent * rand_ratio + latent[rand_idx] * (1 - rand_ratio) # (B, d)
                    batch_dict['rand_idx'] = rand_idx
                    batch_dict['rand_ratio'] = rand_ratio
                else:
                    raise NotImplementedError
                if sample_latent_space_detach:
                    batch_vecs = batch_vecs.detach()
            else:
                batch_vecs = latent # (B, d)

            use_cyc_reg = config.loss.get('use_cyc_reg', False)
            if use_cyc_reg:
                sdf_asap_loss, cyc_loss = self.get_sdf_asap_cyc_loss(batch_vecs, config.loss, batch_dict=batch_dict)
                sdf_cyc_loss = cyc_loss.mean() * config.loss.sdf_cyc_weight
                loss += sdf_cyc_loss
                batch_dict['sdf_cyc_loss'] = sdf_cyc_loss
                state_info['sdf_cyc_loss'] = sdf_cyc_loss.item()
            else:
                sdf_asap_loss = self.get_sdf_asap_loss(batch_vecs, config.loss, batch_dict=batch_dict)
            sdf_asap_loss = sdf_asap_loss.mean() * config.loss.sdf_asap_weight
            loss += sdf_asap_loss
            batch_dict['sdf_asap_loss'] = sdf_asap_loss
            state_info['sdf_asap_loss'] = sdf_asap_loss.item()

        batch_dict["loss"] = loss


    def analysis(self, batch_vecs, batch_dict, config, state_info):
        traces = self.get_sdf_asap_loss(batch_vecs, config.loss, batch_dict=batch_dict)
        batch_dict['traces'] = traces


    def get_induced_vector_field(self, verts, faces, latent, dvec_latent, config):
        """
        Args:
            verts: (n, 3)
            faces: (m, 3)
            latent: (d, )
            dvec_latent: (d, ), the delta_latent
        Returns:
            dvec: (n, 3)
        """
        assert(len(latent.shape) == 1)
        cfg = config.loss
        n = verts.shape[0]
        device = latent.device

        lat_vecs = latent[None, :].repeat(n, 1) # (n, d)

        verts = verts.clone().detach().requires_grad_(True) # (n, 3)
        lat_vecs = lat_vecs.clone().detach().requires_grad_(True) # (n, d)
        iso_sdf_pred = self.decoder(verts, lat_vecs) # (n, 1)

        fx = gradient(iso_sdf_pred, verts) # (n, 3)
        fz = gradient(iso_sdf_pred, lat_vecs) # (n, d)

        with torch.no_grad():
            R, Gz = self.compute_RG(fx, fz, verts, faces, cfg, device) # R: (d, d), Gz: (3*n, d)

            dvec = - Gz @ dvec_latent.reshape(-1, 1) # (3*n_b, 1)
            dvec = dvec.reshape(-1, 3) # (n_b, 3)

        return dvec


    def extract_iso_surface(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        Returns:
            batch_verts_idx: (n1+...+nB, )
            batch_faces_idx: (m1+...+mB, )
            batch_verts: (n1+...+nB, 3)
            batch_faces: (m1+...+mB, 3)
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        batch_verts_idx = []
        batch_faces_idx = []
        batch_verts = []
        batch_faces = []
        for b in range(B):
            x_range, y_range, z_range = cfg.get('x_range', [-1, 1]), cfg.get('y_range', [-0.7, 1.7]), cfg.get('z_range', [-1.1, 0.9])
            verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(self, batch_vecs[b], resolution=cfg.sdf_grid_size, voxel_size=None,
                                                                          max_batch=int(2 ** 18), offset=None, scale=None, points_for_bound=None, verbose=False,
                                                                          x_range=x_range, y_range=y_range, z_range=z_range)
            # denoise mesh, remove small connected components
            split_mesh_list = trimesh.graph.split(trimesh.Trimesh(vertices=verts, faces=faces), only_watertight=False, engine='scipy')
            largest_mesh_idx = np.argmax([split_mesh.vertices.shape[0] for split_mesh in split_mesh_list])
            verts = np.asarray(split_mesh_list[largest_mesh_idx].vertices)
            faces = np.asarray(split_mesh_list[largest_mesh_idx].faces)
            if cfg.get('simplify_mesh', False):
                mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadratic_decimation(2000)
                verts = mesh_sim.vertices
                faces = mesh_sim.faces

            verts = torch.from_numpy(verts).float().to(device)
            faces = torch.from_numpy(faces).long().to(device)

            batch_verts_idx.append(torch.ones_like(verts[:, 0]) * b) # (n_b,)
            batch_faces_idx.append(torch.ones_like(faces[:, 0]) * b) # (m_b,)
            batch_verts.append(verts) # (n_b, 3)
            batch_faces.append(faces) # (m_b, 3)

        batch_verts_idx = torch.cat(batch_verts_idx) # (n1+...+nB)
        batch_faces_idx = torch.cat(batch_faces_idx) # (m1+...+mB)
        batch_verts = torch.cat(batch_verts) # (n1+...+nB, 3)
        batch_faces = torch.cat(batch_faces) # (m1+...+mB, 3)

        return batch_verts_idx, batch_faces_idx, batch_verts, batch_faces


    def get_sdf_asap_loss(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        with torch.no_grad():
            batch_verts_idx, batch_faces_idx, batch_verts, batch_faces = self.extract_iso_surface(batch_vecs, cfg, batch_dict=None)

        batch_vecs_expand = []
        for b in range(B):
            n_b = torch.where(batch_verts_idx == b)[0].shape[0]
            batch_vecs_expand.append(batch_vecs[b:(b+1)].repeat(n_b, 1))
        batch_vecs_expand = torch.cat(batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute gradient XXXXXX
        batch_verts = batch_verts.clone().detach().requires_grad_(True) # (n1+...+nB, 3)
        batch_vecs_expand = batch_vecs_expand.clone().detach().requires_grad_(True) # (n1+...+nB, d)
        iso_sdf_pred = self.decoder(batch_verts, batch_vecs_expand) # (n1+...+NB, 1)

        fx = gradient(iso_sdf_pred, batch_verts) # (n1+...+nB, 3)
        fz = gradient(iso_sdf_pred, batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute regularization loss XXXXXX
        trace_list = []

        for b in range(B):
            batch_verts_mask = (batch_verts_idx == b)
            batch_faces_mask = (batch_faces_idx == b)
            verts_b = batch_verts[batch_verts_mask]
            faces_b = batch_faces[batch_faces_mask]
            n_b = verts_b.shape[0]

            # compute C
            fx_b = fx[batch_verts_mask] # (n_b, 3)
            C0, C1, C_vals = [], [], []
            lin_b = torch.arange(n_b, device=device)
            C0 = torch.stack((lin_b, lin_b, lin_b), dim=0).T.reshape(-1)
            C1 = torch.stack((lin_b * 3, lin_b * 3 + 1, lin_b * 3 + 2), dim=0).T.reshape(-1)
            C_vals = fx_b.reshape(-1)
            C_indices, C_vals = ts.coalesce([C0, C1], C_vals, n_b, n_b * 3)
            C = torch.sparse_coo_tensor(C_indices, C_vals, (n_b, 3*n_b))

            # compute F
            F = fz[batch_verts_mask] # (n_b, d)

            hessian_b = compute_asap3d_sparse(verts_b, faces_b, weight_asap=cfg.weight_asap) # (3*n_b, 3*n_b), sparse
            hessian_b = hessian_b.float()

            implicit_reg_type = cfg.get('implicit_reg_type', None)
            if implicit_reg_type == 'dense_inverse':
                hessian_b = hessian_b.to_dense()
                hessian_b = hessian_b + cfg.mu_asap * torch.eye(n_b * 3, device=device) if cfg.get('add_mu_diag_to_hessian', True) else hessian_b

                hessian_b_pinv = torch.linalg.inv(hessian_b)
                hessian_b_pinv = (hessian_b_pinv + hessian_b_pinv.T) / 2.0 # hessian_b_pinv is symmetric

                CH = ts.spmm(C_indices, C_vals, n_b, n_b * 3, hessian_b_pinv) # (n_b, 3*n_b)
                CHCT = ts.spmm(C_indices, C_vals, n_b, n_b * 3, CH.T) # (n_b, n_b)
                CHCT = (CHCT + CHCT.T) / 2
                CHCT = CHCT + cfg.mu_asap * torch.eye(n_b, device=device) # some row of C might be 0

                CHCT_inv = torch.linalg.inv(CHCT)
                CHCT_inv = (CHCT_inv + CHCT_inv.T) / 2

                R = F.T @ CHCT_inv @ F
            else:
                raise NotImplementedError

            e = torch.linalg.eigvalsh(R).clamp(0)
            e = e ** 0.5
            trace = e.sum()
            trace_list.append(trace)

        traces = torch.stack(trace_list)
        return traces


    def compute_RG(self, fx_b, fz_b, verts_b, faces_b, cfg, device):
        '''
        fx_b: (n_b, 3)
        fz_b: (n_b, d)
        R: (d, d)
        Gz: (3*n_b, d)
        '''
        n_b = fx_b.shape[0]
        # compute C
        C0, C1, C_vals = [], [], []
        lin_b = torch.arange(n_b, device=device)
        C0 = torch.stack((lin_b, lin_b, lin_b), dim=0).T.reshape(-1)
        C1 = torch.stack((lin_b * 3, lin_b * 3 + 1, lin_b * 3 + 2), dim=0).T.reshape(-1)
        C_vals = fx_b.reshape(-1)
        C_indices, C_vals = ts.coalesce([C0, C1], C_vals, n_b, n_b * 3)
        C = torch.sparse_coo_tensor(C_indices, C_vals, (n_b, 3*n_b))

        # compute F
        F = fz_b # (n_b, d)

        # compute hessian
        hessian_b = compute_asap3d_sparse(verts_b, faces_b, weight_asap=cfg.weight_asap) # (3*n_b, 3*n_b), sparse
        hessian_b = hessian_b.float()

        implicit_reg_type = cfg.get('implicit_reg_type', None)
        if implicit_reg_type == 'dense_inverse':
            hessian_b = hessian_b.to_dense()
            hessian_b = hessian_b + cfg.mu_asap * torch.eye(n_b * 3, device=device) if cfg.get('add_mu_diag_to_hessian', True) else hessian_b

            hessian_b_pinv = torch.linalg.inv(hessian_b)
            hessian_b_pinv = (hessian_b_pinv + hessian_b_pinv.T) / 2.0 # hessian_b_pinv is symmetric

            CH = ts.spmm(C_indices, C_vals, n_b, n_b * 3, hessian_b_pinv) # (n_b, 3*n_b)
            CHCT = ts.spmm(C_indices, C_vals, n_b, n_b * 3, CH.T) # (n_b, n_b)
            CHCT = (CHCT + CHCT.T) / 2
            CHCT = CHCT + cfg.mu_asap * torch.eye(n_b, device=device) # some row of C might be 0

            CHCT_inv = torch.linalg.inv(CHCT)
            CHCT_inv = (CHCT_inv + CHCT_inv.T) / 2

            CHCT_invF = CHCT_inv @ F # (n_b, d)

            R = F.T @ CHCT_invF

            Gz = CH.T @ CHCT_invF # (3*n_b, d)
        else:
            raise NotImplementedError

        return R, Gz


    def get_sdf_asap_cyc_loss(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        with torch.no_grad():
            batch_verts_idx, batch_faces_idx, batch_verts, batch_faces = self.extract_iso_surface(batch_vecs, cfg, batch_dict=None)

        batch_vecs_expand = []
        for b in range(B):
            n_b = torch.where(batch_verts_idx == b)[0].shape[0]
            batch_vecs_expand.append(batch_vecs[b:(b+1)].repeat(n_b, 1))
        batch_vecs_expand = torch.cat(batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute gradient XXXXXX
        batch_verts = batch_verts.clone().detach().requires_grad_(True) # (n1+...+nB, 3)
        batch_vecs_expand = batch_vecs_expand.clone().detach().requires_grad_(True) # (n1+...+nB, d)
        iso_sdf_pred = self.decoder(batch_verts, batch_vecs_expand) # (n1+...+NB, 1)

        fx = gradient(iso_sdf_pred, batch_verts) # (n1+...+nB, 3)
        fz = gradient(iso_sdf_pred, batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute regularization loss XXXXXX
        trace_list = []
        cyc_list = []

        rand_dim_idx = np.random.choice(batch_vecs_expand.shape[-1], size=(1,))
        sync_batch_rand_dim_idx = cfg.get('sync_batch_rand_dim_idx', False)
        for b in range(B):
            batch_verts_mask = (batch_verts_idx == b)
            batch_faces_mask = (batch_faces_idx == b)
            verts_b = batch_verts[batch_verts_mask]
            faces_b = batch_faces[batch_faces_mask]
            lat_vecs_b = batch_vecs_expand[batch_verts_mask] # (n_b, d)
            n_b = verts_b.shape[0]

            R, Gz = self.compute_RG(fx[batch_verts_mask], fz[batch_verts_mask], verts_b, faces_b, cfg, device)

            ################# Compute cyc loss START #################
            # sample v
            edz = torch.zeros_like(lat_vecs_b) # (n_b, d)
            if not sync_batch_rand_dim_idx:
                rand_dim_idx = np.random.choice(lat_vecs_b.shape[-1], size=(1,))
            edz[:, rand_dim_idx] = 1
            edz = edz * cfg.eps_cyc
            lat_vecs_b_new = lat_vecs_b + edz # (n_b, d)

            dvec = - Gz @ edz[0].reshape(-1, 1) # (3*n_b, 1)
            dvec = dvec.reshape(-1, 3) # (n_b, 3)

            verts_b_new = verts_b + dvec # (n_b, 3)

            # compute gradient new
            verts_b_new = verts_b_new.clone().detach().requires_grad_(True) # (n_b, 3)
            lat_vecs_b_new = lat_vecs_b_new.clone().detach().requires_grad_(True) # (n_b, d)
            iso_sdf_pred_new = self.decoder(verts_b_new, lat_vecs_b_new) # (n_b, 1)

            fx_b_new = gradient(iso_sdf_pred_new, verts_b_new) # (n_b, 3)
            fz_b_new = gradient(iso_sdf_pred_new, lat_vecs_b_new) # (n_b, d)

            # compute C
            R_new, Gz_new = self.compute_RG(fx_b_new, fz_b_new, verts_b_new, faces_b, cfg, device)

            cyc_diff = (Gz_new - Gz) / cfg.eps_cyc
            e_cyc = torch.mean(cyc_diff * cyc_diff)
            ################# Compute cyc loss END #################

            e = torch.linalg.eigvalsh(R).clamp(0)
            e = e ** 0.5
            trace = e.sum()
            trace_list.append(trace)

            cyc_list.append(e_cyc)

        traces = torch.stack(trace_list)
        cyc_loss = torch.stack(cyc_list)
        return traces, cyc_loss



