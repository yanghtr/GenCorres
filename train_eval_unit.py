import os, sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from loguru import logger

from pyutils import get_directory, to_device
from utils import implicit_utils
from utils.geom_utils import embedded_deformation

import point_cloud_utils as pcu


def save_obj(fname, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(fname)


# Only set requires_grad is NOT enough to freeze params when using shared optimizer
# Ref: https://discuss.pytorch.org/t/parameters-with-requires-grad-false-are-updated-during-training/90096/9
def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad
        if requires_grad:
            param.grad = torch.zeros_like(param)
        else:
            param.grad = None


def set_params_grad(model, lat_vecs, config):
    model_module = model.module if config.dist_train else model 
    set_requires_grad(model_module.decoder, True)
    set_requires_grad(lat_vecs, True)


def train_one_epoch(state_info, config, train_loader, model, lat_vecs, optimizer_train, writer, latents_all=None):
    # set_params_grad(model, lat_vecs, config)
    model.train()

    epoch = state_info['epoch']
    device = state_info['device']

    # ASAP
    if config.local_rank == 0 and (config.use_sdf_asap or config.use_mesh_arap):
        logger.warning("use ARAP/ASAP loss")

    for b, batch_dict in enumerate(train_loader):
        state_info['b'] = b
        optimizer_train.zero_grad()
        batch_dict = to_device(batch_dict, device)

        batch_vecs = None
        if lat_vecs is not None:
            batch_vecs = lat_vecs(batch_dict['idx']) # (B, latent_dim)
        if latents_all is not None:
            batch_dict['latents_all'] = latents_all

        batch_dict = model(batch_vecs, batch_dict, config, state_info) # (B, N, 3)
        batch_dict.update({k : v.mean() for k, v in batch_dict.items() if 'loss' in k})
        state_info.update({k : v.item() for k, v in batch_dict.items() if 'loss' in k})

        loss = batch_dict["loss"]
        loss.backward()
        optimizer_train.step()

        if config.local_rank == 0 and b % config.log.log_batch_interval == 0:
            global_step = (state_info['epoch'] * state_info['len_train_loader'] + b ) * config.optimization[config.rep].batch_size
            writer.log_state_info(state_info)
            writer.log_summary(state_info, global_step, mode='train')

    return state_info


def test_opt_one_epoch(state_info, config, test_opt_loader, model, test_lat_vecs, optimizer_test, writer):
    model.eval() # NOTE: only test_lat_vecs are optimized

    epoch = state_info['epoch']
    device = state_info['device']

    for b, batch_dict in enumerate(test_opt_loader):
        state_info['b'] = b
        optimizer_test.zero_grad()
        batch_dict = to_device(batch_dict, device)

        batch_vecs = test_lat_vecs(batch_dict['idx']) # (B, latent_dim)

        batch_dict = model(batch_vecs, batch_dict, config, state_info, only_decoder_forward=True) # (B, N, 3)
        batch_dict.update({k : v.mean() for k, v in batch_dict.items() if 'loss' in k})
        state_info.update({k : v.item() for k, v in batch_dict.items() if 'loss' in k})

        loss = batch_dict["loss"]
        loss.backward()
        optimizer_test.step()

        if config.local_rank == 0 and b % config.log.log_batch_interval == 0:
            global_step = (state_info['epoch'] * state_info['len_test_opt_loader'] + b ) * config.optimization[config.rep].batch_size
            writer.log_state_info(state_info)
            writer.log_summary(state_info, global_step, mode='test')

    return state_info


def recon_from_lat_vecs(state_info, config, recon_loader, model, recon_lat_vecs, results_dir):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    device = state_info['device']

    if config.rep == 'sdf':
        logger.info(" reconstruct mesh from sdf predicted by sdfnet ")

        parallel_idx_list = range(len(recon_loader.dataset))
        if config.parallel_idx >= 0:
            parallel_idx_list = parallel_idx_list[config.parallel_idx * config.parallel_interval : (config.parallel_idx + 1) * config.parallel_interval]
        logger.info(f"number of recon mesh: {len(parallel_idx_list)} in {len(recon_loader.dataset)}")

        for i, batch_dict in enumerate(recon_loader):
            if i not in parallel_idx_list:
                continue
            fid = recon_loader.dataset.fid_list[i]
            print(f"i={i}, fid={fid}")
            batch_dict = to_device(batch_dict, device)
            
            if config.auto_decoder:
                # get latent_vec from recon_lat_vecs
                assert(recon_lat_vecs is not None)
                latent_vec = recon_lat_vecs(batch_dict['idx']) # (B, latent_dim)
            else:
                assert(recon_lat_vecs is None)
                latent_vec, _, _ = model(None, batch_dict, config, None, only_encoder_forward=True)
            assert(latent_vec.shape[0] == 1) # batch_size == 1
            latent_vec = latent_vec[0]

            points_for_bound = batch_dict['points_mnfld'][0] if 'points_mnfld' in batch_dict else None
            x_range, y_range, z_range = config.loss.get('x_range', [-1, 1]), config.loss.get('y_range', [-0.7, 1.7]), config.loss.get('z_range', [-1.1, 0.9])
            # verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(model, latent_vec, resolution=128, voxel_size=None, max_batch=int(2 ** 18), offset=None, scale=None, points_for_bound=points_for_bound)
            verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(model, latent_vec, resolution=128, voxel_size=None, max_batch=int(2 ** 17), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)

            save_obj(f"{results_dir}/{fid}.obj", verts, faces)
            with open(f"{results_dir}/{fid}.pkl", "wb") as f:
                dump_dict = {'points_mnfld': batch_dict['points_mnfld'][0].detach().cpu().numpy(),
                             'samples_nonmnfld': batch_dict['samples_nonmnfld'][0].detach().cpu().numpy(),
                             'recon_verts': verts, 'recon_faces': faces}
                if 'raw_mesh_verts' in batch_dict:
                    dump_dict.update({'raw_mesh_verts': batch_dict['raw_mesh_verts'][0].detach().cpu().numpy()})
                if 'raw_mesh_faces' in batch_dict:
                    dump_dict.update({'raw_mesh_faces': batch_dict['raw_mesh_faces'][0].detach().cpu().numpy()})
                pickle.dump(dump_dict, f)

    elif config.rep == 'mesh':
        logger.info(" reconstruct mesh from mesh predicted by meshnet ")

        parallel_idx_list = range(len(recon_loader.dataset))
        if config.parallel_idx >= 0:
            parallel_idx_list = parallel_idx_list[config.parallel_idx * config.parallel_interval : (config.parallel_idx + 1) * config.parallel_interval]
        logger.info(f"number of recon mesh: {len(parallel_idx_list)} in {len(recon_loader.dataset)}")

        for i, batch_dict in enumerate(recon_loader):
            if i not in parallel_idx_list:
                continue
            fid = recon_loader.dataset.fid_list[i]
            print(f"i={i}, fid={fid}")
            
            if config.auto_decoder:
                assert(recon_lat_vecs is not None)
                batch_dict = to_device(batch_dict, device)
                latent_vec = recon_lat_vecs(batch_dict['idx'])
                assert(latent_vec.shape[0] == 1) # batch_size == 1
                batch_dict = model(latent_vec, batch_dict, config, state_info=None)
            else:
                raise NotImplementedError

            verts_init = batch_dict['verts_init_nml'][0].detach().cpu().numpy() * model.data_std_gpu.numpy() + model.data_mean_gpu.numpy()
            verts_pred = batch_dict['mesh_verts_nml_pred'][0].detach().cpu().numpy() * model.data_std_gpu.numpy() + model.data_mean_gpu.numpy()
            verts_raw = batch_dict['verts_raw'][0].detach().cpu().numpy()
            faces_raw = batch_dict['faces_raw'][0].detach().cpu().numpy()
            template_faces = recon_loader.dataset.template_faces

            save_obj(f"{results_dir}/{fid}.obj", verts_pred, template_faces)
            with open(f"{results_dir}/{fid}.pkl", "wb") as f:
                dump_dict = {'verts_init': verts_init, 'verts_raw': verts_raw, 'faces_raw': faces_raw,
                             'verts_pred': verts_pred, 'template_faces': template_faces}
                pickle.dump(dump_dict, f)
    else:
        raise NotImplementedError


def interp_from_edges(state_info, config, interp_loader, model, latents_all, edge_ids, results_dir):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    device = state_info['device']
    num_interp = 10

    logger.info(" interpolate sdf predicted by sdfnet ")

    if config.rep == 'sdf':
        logger.info(" reconstruct mesh from sdf predicted by sdfnet ")

        parallel_idx_list = range(edge_ids.shape[0])
        if config.parallel_idx >= 0:
            parallel_idx_list = parallel_idx_list[config.parallel_idx * config.parallel_interval : (config.parallel_idx + 1) * config.parallel_interval]

        for eid in parallel_idx_list:
            src_idx, tgt_idx = edge_ids[eid]
            src_fid = interp_loader.dataset.fid_list[src_idx]
            tgt_fid = interp_loader.dataset.fid_list[tgt_idx]
            latent_src = latents_all[src_idx]
            latent_tgt = latents_all[tgt_idx]
            logger.info(f"interpolate {eid}: {src_fid} ({src_idx}-th) and {tgt_fid} ({tgt_idx}-th)")

            dump_dir = f"{results_dir}/{src_idx}_{tgt_idx}"
            if os.path.exists(dump_dir):
                continue
            dump_dir = get_directory( dump_dir )
            for i_interp in range(0, num_interp + 1): 
                ri = i_interp / num_interp

                latent_interp = latent_src * (1 - ri) + latent_tgt * ri

                x_range, y_range, z_range = config.loss.get('x_range', [-1, 1]), config.loss.get('y_range', [-0.7, 1.7]), config.loss.get('z_range', [-1.1, 0.9])
                # verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(model, latent_interp, resolution=256, max_batch=int(2 ** 18), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)
                verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(model, latent_interp, resolution=128, max_batch=int(2 ** 17), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)
                mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadratic_decimation(2000)
                verts = mesh_sim.vertices
                faces = mesh_sim.faces

                save_obj(f"{dump_dir}/{src_idx}_{tgt_idx}_{i_interp:02d}.obj", verts, faces)


def interp_from_lat_vecs(state_info, config, interp_loader, model, interp_lat_vecs, results_dir):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    device = state_info['device']
    num_interp = 10

    if config.split == 'train':
        src_fid = config.get('interp_src_fid', '50022-knees-knees.001582')
        tgt_fid = config.get('interp_tgt_fid', '50022-knees-knees.002101')
    elif config.split == 'test':
        src_fid = config.get('interp_src_fid', '50009-running_on_spot-running_on_spot.000366')
        tgt_fid = config.get('interp_tgt_fid', '50002-chicken_wings-chicken_wings.004011')
    else:
        raise NotImplementedError

    logger.info(" interpolate sdf predicted by sdfnet ")

    for i, batch_dict in enumerate(interp_loader):
        fid = interp_loader.dataset.fid_list[batch_dict['idx'][0]]
        if fid not in [src_fid, tgt_fid]:
            continue
        
        batch_dict = to_device(batch_dict, device)
        if config.auto_decoder:
            assert(interp_lat_vecs is not None)
            latent_vec = interp_lat_vecs(batch_dict['idx']) # (B, latent_dim)
        else:
            assert(interp_lat_vecs is None)
            latent_vec, _, _ = model(None, batch_dict, config, None, only_encoder_forward=True)

        assert(latent_vec.shape[0] == 1) # batch_size == 1
        latent_vec = latent_vec[0]
        if fid == src_fid:
            latent_src = latent_vec
            src_idx = i
        if fid == tgt_fid:
            latent_tgt = latent_vec
            tgt_idx = i

    logger.info(f"interpolate {src_fid} ({src_idx}-th) and {tgt_fid} ({tgt_idx}-th)")
    for i_interp in range(0, num_interp + 1): 
        ri = i_interp / num_interp

        latent_interp = latent_src * (1 - ri) + latent_tgt * ri
        # DEBUG START
        # norm_diff = torch.linalg.norm(latent_src - latent_tgt)
        # latent_interp += torch.randn_like(latent_interp) * norm_diff * config.loss.line_gau_scale
        # DEBUG END

        dump_dir = get_directory( f"{results_dir}/{src_idx}_{tgt_idx}" )
        x_range, y_range, z_range = config.loss.get('x_range', [-1, 1]), config.loss.get('y_range', [-0.7, 1.7]), config.loss.get('z_range', [-1.1, 0.9])
        verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(model, latent_interp, resolution=128, max_batch=int(2 ** 17), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)
        # mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadratic_decimation(2000)
        # verts = mesh_sim.vertices
        # faces = mesh_sim.faces

        save_obj(f"{dump_dir}/{src_idx}_{tgt_idx}_{i_interp:02d}.obj", verts, faces)

    def _copy_raw_mesh(_fid, _idx):
        _fname = '/'.join(_fid.split('-'))
        _fpath = f"{interp_loader.dataset.raw_mesh_dir}/{_fname}.{interp_loader.dataset.raw_mesh_file_type}"
        os.system(f"cp {_fpath} ./{dump_dir}/{_idx}.{interp_loader.dataset.raw_mesh_file_type}")

    _copy_raw_mesh(src_fid, src_idx)
    _copy_raw_mesh(tgt_fid, tgt_idx)


def interp_corres_from_lat_vecs(state_info, config, interp_loader, model, interp_lat_vecs, results_dir):
    '''
    Args:
        interp_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    device = state_info['device']
    num_interp = 30

    if config.split == 'train':
        src_fid = config.get('interp_src_fid', '50022-knees-knees.001582')
        tgt_fid = config.get('interp_tgt_fid', '50022-knees-knees.002101')
    elif config.split == 'test':
        src_fid = config.get('interp_src_fid', '50009-running_on_spot-running_on_spot.000366')
        tgt_fid = config.get('interp_tgt_fid', '50002-chicken_wings-chicken_wings.004011')
    else:
        raise NotImplementedError

    logger.info(" interpolate sdf predicted by sdfnet and propagate corres")

    for i, batch_dict in enumerate(interp_loader):
        fid = interp_loader.dataset.fid_list[batch_dict['idx'][0]]
        if fid not in [src_fid, tgt_fid]:
            continue
        
        batch_dict = to_device(batch_dict, device)
        if config.auto_decoder:
            assert(interp_lat_vecs is not None)
            latent_vec = interp_lat_vecs(batch_dict['idx']) # (B, latent_dim)
        else:
            assert(interp_lat_vecs is None)
            latent_vec, _, _ = model(None, batch_dict, config, None, only_encoder_forward=True)

        assert(latent_vec.shape[0] == 1) # batch_size == 1
        latent_vec = latent_vec[0]
        if fid == src_fid:
            latent_src = latent_vec
            src_idx = i
            raw_mesh_verts_src = batch_dict['raw_mesh_verts'][0].detach().cpu().numpy()
            raw_mesh_faces_src = batch_dict['raw_mesh_faces'][0].detach().cpu().numpy()
        if fid == tgt_fid:
            latent_tgt = latent_vec
            tgt_idx = i
            raw_mesh_verts_tgt = batch_dict['raw_mesh_verts'][0].detach().cpu().numpy()
            raw_mesh_faces_tgt = batch_dict['raw_mesh_faces'][0].detach().cpu().numpy()

    dump_dir = get_directory( f"{results_dir}/{src_idx}_{tgt_idx}" )
    logger.info(f"Propagate corres start: {src_fid} ({src_idx}-th) and {tgt_fid} ({tgt_idx}-th)")

    mesh_src = trimesh.Trimesh(vertices=raw_mesh_verts_src, faces=raw_mesh_faces_src, process=False)
    mesh_tgt = trimesh.Trimesh(vertices=raw_mesh_verts_tgt, faces=raw_mesh_faces_tgt, process=False)
    mesh_src_sim = mesh_src.simplify_quadratic_decimation(2000)
    mesh_tgt_sim = mesh_tgt.simplify_quadratic_decimation(2000)

    mesh_src.export(f"{dump_dir}/{src_idx}.obj")
    mesh_tgt.export(f"{dump_dir}/{tgt_idx}.obj")
    mesh_src_sim.export(f"{dump_dir}/{src_idx}_sim.obj")
    mesh_tgt_sim.export(f"{dump_dir}/{tgt_idx}_sim.obj")

    mesh_temp = mesh_src
    mesh_temp_sim = mesh_src_sim
    verts_temp = mesh_temp_sim.vertices.astype(np.float32)
    faces_temp = mesh_temp_sim.faces
    for i_interp in range(0, num_interp): # exclude target
        ri = i_interp / num_interp

        latent_interp = latent_src * (1 - ri) + latent_tgt * ri # (latent_dim,)
        dvec_latent = (latent_tgt - latent_src) / num_interp

        verts_temp = torch.from_numpy(verts_temp).to(latent_interp.device)
        faces_temp = torch.from_numpy(faces_temp).to(latent_interp.device)

        verts_dvec = model.get_induced_vector_field(verts_temp, faces_temp, latent_interp, dvec_latent, config)
        verts_temp = verts_temp + verts_dvec

        # project verts_temp to interpolated_shape
        verts_temp = verts_temp.detach().cpu().numpy()
        faces_temp = faces_temp.detach().cpu().numpy()
        save_obj(f"{dump_dir}/{src_idx}_{tgt_idx}_{i_interp:02d}_prop.obj", verts_temp, faces_temp)

    mesh_temp_sim_def = trimesh.Trimesh(vertices=verts_temp, faces=faces_temp, process=False) # NOTE: have to set process=False

    mesh_temp_def = embedded_deformation(mesh_temp, mesh_temp_sim, mesh_temp_sim_def)

    # compute correspondence between mesh_temp_def and mesh_tgt
    dists_def_to_tgt, corres_def_to_tgt = pcu.k_nearest_neighbors(mesh_temp_def.vertices, mesh_tgt.vertices, k=1) # same shape as src
    dists_tgt_to_def, corres_tgt_to_def = pcu.k_nearest_neighbors(mesh_tgt.vertices, mesh_temp_def.vertices, k=1) # same shape as tgt

    # compute error
    assert(np.all(mesh_temp_def.faces == mesh_tgt.faces))
    assert(corres_def_to_tgt.shape == corres_tgt_to_def.shape) # NOTE: here ONLY applies to DFAUST: same topology in evaluation
    logger.info(f"mean corres error: {tgt_fid} ({tgt_idx}-th) -> {src_fid} ({src_idx}-th): {np.mean(dists_tgt_to_def)} ")
    logger.info(f"mean corres error: {src_fid} ({src_idx}-th) -> {tgt_fid} ({tgt_idx}-th): {np.mean(dists_def_to_tgt)} ")

    mesh_temp_def.export(f"{dump_dir}/{src_idx}_{tgt_idx}_def.obj")
    np.save(f"{dump_dir}/{src_idx}_{tgt_idx}_corres.npy", corres_def_to_tgt)
    np.save(f"{dump_dir}/{tgt_idx}_{src_idx}_corres.npy", corres_tgt_to_def)
    np.save(f"{dump_dir}/{src_idx}_{tgt_idx}_dists.npy", dists_def_to_tgt)
    np.save(f"{dump_dir}/{tgt_idx}_{src_idx}_dists.npy", dists_tgt_to_def)
    print('Done')


def analysis_one_epoch(state_info, config, mesh_sdf_loader, model, lat_vecs, results_dir):
    '''
    Args:
        lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    epoch = state_info['epoch']
    device = state_info['device']

    if config.rep == 'sdf':
        latents_all_dict = {}
        for b, batch_dict in enumerate(mesh_sdf_loader):
            print(b)
            if config.auto_decoder:
                # get latent_vec from lat_vecs
                raise NotImplementedError
            else:
                assert(lat_vecs is None)
                batch_dict = to_device(batch_dict, device)
                latent_vec, _, _ = model(None, batch_dict, config, None, only_encoder_forward=True)

            latents_all_dict.update({
                idx.item(): {
                    'fid': mesh_sdf_loader.dataset.fid_list[idx],
                    'latent': latent_vec.detach().cpu().numpy()[ii]
                } for ii, idx in enumerate(batch_dict['idx'])
            })
        with open(f"{results_dir}/latents_all_{config.split}_{epoch}.pkl", 'wb') as f:
            pickle.dump(latents_all_dict, f)

        latents_all = np.zeros((len(latents_all_dict), config.latent_dim))
        for k, v in latents_all_dict.items():
            latents_all[k] = v['latent']
        np.save(f"{results_dir}/latents_all_{config.split}_{epoch}.npy", latents_all)
    else:
        assert(config.rep in ['mesh', 'all'])
        raise NotImplementedError


def evaluate(state_info, config, eval_loader, model, eval_lat_vecs, results_dir):
    model.eval()
    device = state_info['device']
    epoch = state_info['epoch']

    if config.rep == 'sdf':
        parallel_idx_list = range(len(eval_loader.dataset))
        if config.parallel_idx >= 0:
            parallel_idx_list = parallel_idx_list[config.parallel_idx * config.parallel_interval : (config.parallel_idx + 1) * config.parallel_interval]
        logger.info(f"number of eval mesh: {len(eval_loader.dataset)}")

        dis_cd_dict = {}
        for i, batch_dict in enumerate(eval_loader):
            if i not in parallel_idx_list:
                continue
            fid = eval_loader.dataset.fid_list[i]
            print(f"i={i}, fid={fid}")
            batch_dict = to_device(batch_dict, device)
            
            if config.auto_decoder:
                # get latent_vec from eval_lat_vecs
                assert(eval_lat_vecs is not None)
                latent_vec = eval_lat_vecs(batch_dict['idx']) # (B, latent_dim)
            else:
                assert(eval_lat_vecs is None)
                latent_vec, _, _ = model(None, batch_dict, config, None, only_encoder_forward=True)
            assert(latent_vec.shape[0] == 1) # batch_size == 1
            latent_vec = latent_vec[0]

            x_range, y_range, z_range = config.loss.get('x_range', [-1, 1]), config.loss.get('y_range', [-0.7, 1.7]), config.loss.get('z_range', [-1.1, 0.9])
            verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(model, latent_vec, resolution=256, voxel_size=None, max_batch=int(2 ** 17), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)

            # save_obj(f"{results_dir}/{fid}.obj", verts, faces)
            assert('raw_mesh_verts' in batch_dict and 'raw_mesh_faces' in batch_dict)
            mesh_gt = trimesh.Trimesh(vertices=batch_dict['raw_mesh_verts'][0].detach().cpu().numpy(), faces=batch_dict['raw_mesh_faces'][0].detach().cpu().numpy())
            mesh_pred = trimesh.Trimesh(vertices=verts, faces=faces)
            pc_gt = trimesh.sample.sample_surface(mesh_gt, 30000)[0]
            pc_pred = trimesh.sample.sample_surface(mesh_pred, 30000)[0]
            dis_cd = pcu.chamfer_distance(pc_pred, pc_gt) # symmetric and bidirectional, d1.mean() + d2.mean()
            dis_cd_dict.update({ i: {'fid': fid, 'dis_cd': dis_cd}})
            # np.mean([v['dis_cd'] for k, v in dis_cd_dict.items()])

        with open(f"{results_dir}/eval_{epoch}_{config.parallel_idx:03d}.pkl", 'wb') as f:
            pickle.dump(dis_cd_dict, f)

    elif config.rep == 'mesh':
        parallel_idx_list = range(len(eval_loader.dataset))
        if config.parallel_idx >= 0:
            parallel_idx_list = parallel_idx_list[config.parallel_idx * config.parallel_interval : (config.parallel_idx + 1) * config.parallel_interval]
        logger.info(f"number of eval mesh: {len(parallel_idx_list)} in {len(eval_loader.dataset)}")

        for i, batch_dict in enumerate(eval_loader):
            if i not in parallel_idx_list:
                continue
            fid = eval_loader.dataset.fid_list[i]
            print(f"i={i}, fid={fid}")
            
            if config.auto_decoder:
                assert(eval_lat_vecs is not None)
                batch_dict = to_device(batch_dict, device)
                latent_vec = eval_lat_vecs(batch_dict['idx'])
                assert(latent_vec.shape[0] == 1) # batch_size == 1
                batch_dict = model(latent_vec, batch_dict, config, state_info=None)
            else:
                raise NotImplementedError

            verts_pred = batch_dict['mesh_verts_nml_pred'][0].detach().cpu().numpy() * model.data_std_gpu.numpy() + model.data_mean_gpu.numpy()
            verts_raw = batch_dict['verts_raw'][0].detach().cpu().numpy()

            dists_def_to_tgt, corres_def_to_tgt = pcu.k_nearest_neighbors(verts_pred, verts_raw, k=1)
            dists_tgt_to_def, corres_tgt_to_def = pcu.k_nearest_neighbors(verts_raw, verts_pred, k=1)
            np.save(f"{results_dir}/{fid}_d2t_corres.npy", corres_def_to_tgt)
            np.save(f"{results_dir}/{fid}_t2d_corres.npy", corres_tgt_to_def)
    else:
        raise NotImplementedError







