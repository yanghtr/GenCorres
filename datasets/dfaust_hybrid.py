import os
import os.path as osp
import glob
import json
import pickle
from collections import defaultdict

import torch
import trimesh
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from loguru import logger
from sklearn.decomposition import PCA


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


class DFaustHybridDataSet(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 config,
                 **kwargs):
        '''
        Args:
            sdf_dir: raw sdf dir
            raw_mesh_dir: raw mesh dir, might not have consistent topology
            registration_dir: registered mesh dir, must have consistent topology
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.config = config
        self.mode = mode
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else:
            raise ValueError('invalid mode')

        self.data_dir = config.data_dir
        self.sdf_dir = config.sdf_dir
        self.raw_mesh_dir = config.raw_mesh_dir
        self.registration_dir = config.registration_dir
        self.num_samples = config.num_samples
        self.template_path = config.template_path

        # load data split
        split_cfg_fname = config.split_cfg[split]
        current_dir = os.path.dirname(os.path.realpath(__file__))
        split_path = f"{current_dir}/splits/dfaust_hybrid/{split_cfg_fname}"
        with open(split_path, "r") as f:
            split_names = json.load(f)

        self.fid_list = self.get_fid_list(split_names)
        self.num_data = len(self.fid_list)

        self.raw_mesh_file_type = config.get('raw_mesh_file_type', 'ply')
        logger.info(f"dataset mode = {mode}, split = {split}, len = {self.num_data}\n")

        # load temlate mesh for meshnet. Share topology. NOTE: used for meshnet, different from temp(late) in sdfnet
        template_mesh = trimesh.load(self.template_path, process=False, maintain_order=True)
        self.template_points = torch.from_numpy(template_mesh.vertices)
        self.template_faces = np.asarray(template_mesh.faces)
        self.num_nodes = self.template_points.shape[0]

        # load sim mesh data if exists
        self.sim_mesh_dir = config.get('sim_mesh_dir', None)
        if self.sim_mesh_dir is not None:
            self.verts_sim_list = []
            self.faces_sim_list = []
            for fid in self.fid_list:
                fname = '/'.join(fid.split('-'))
                sim_mesh_pkl = pickle.load(open(f"{self.sim_mesh_dir}/{fname}_sim.pkl", 'rb'))
                self.verts_sim_list.append(sim_mesh_pkl['verts_sim'].astype(np.float32))
                self.faces_sim_list.append(sim_mesh_pkl['faces_sim'])

        # load init mesh data if exists
        self.init_mesh_dir = config.get('init_mesh_dir', None)
        if self.init_mesh_dir is not None:
            verts_init_list = []
            for fid in tqdm(self.fid_list):
                mesh_init = trimesh.load(f"{self.init_mesh_dir}/{fid}_init.obj", process=False, maintain_order=True)
                verts_init_list.append(mesh_init.vertices.astype(np.float32))
            self.verts_init = np.stack(verts_init_list) # (1000, 6890, 3)
            print(f'Finish loading verts_init, shape = {self.verts_init.shape}')
            assert(self.verts_init.shape[0] == self.num_data)
            self.mean_init = self.verts_init.mean(axis=0) # only verts_init always has consistent correspondence
            self.std_init = self.verts_init.std(axis=0)
            # IMPORTANT TODO: if SMAL, set self.std_init = 0.2

            # Normalize mesh data
            # NOTE: the target of the prediction: verts_init is normalized
            self.verts_init_nml = (self.verts_init - self.mean_init) / self.std_init

            self.use_vert_pca = config.get('use_vert_pca', True)
            self.pca = PCA(n_components=config.pca_n_comp)
            self.pca.fit(self.verts_init_nml.reshape(self.num_data, -1))
            self.pca_axes = self.pca.components_
            pca_sv = np.matmul(self.verts_init_nml.reshape(self.num_data, -1), self.pca_axes.transpose())
            self.pca_sv_mean = np.mean(pca_sv, axis=0)
            self.pca_sv_std = np.std(pca_sv, axis=0)
            print(f'Finish computing PCA')

        # load raw mesh
        if self.rep == 'mesh':
            self.verts_raw_list = []
            self.faces_raw_list = []
            for fid in self.fid_list:
                fname = '/'.join(fid.split('-'))
                mesh_raw = trimesh.load(f"{self.raw_mesh_dir}/{fname}.{self.raw_mesh_file_type}", process=False, maintain_order=True)
                self.verts_raw_list.append(mesh_raw.vertices.astype(np.float32))
                self.faces_raw_list.append(mesh_raw.faces)


    def get_fid_list(self, split_names):
        fid_list = []
        assert(len(split_names) == 1)
        for dataset in split_names:
            for class_name in split_names[dataset]:
                for instance_name in split_names[dataset][class_name]:
                    for shape in split_names[dataset][class_name][instance_name]:
                        fid = f"{class_name}-{instance_name}-{shape}"
                        fid_list.append(fid)
        return fid_list


    def update_pca_sv(self, train_pca_axes, train_pca_sv_mean, train_pca_sv_std):
        pca_sv = np.matmul(self.verts_init_nml.reshape(self.num_data, -1), train_pca_axes.transpose())
        self.pca_sv = (pca_sv - train_pca_sv_mean) / train_pca_sv_std


    def __len__(self):
        return self.num_data


    def __getitem__(self, idx):
        data_dict = {}
        data_dict['idx'] = torch.tensor(idx, dtype=torch.long)
        fid = self.fid_list[idx]
        fname = '/'.join(fid.split('-'))

        if self.rep in ['mesh']:
            # no sdf, only load mesh. TODO: verts num diff, need to use PyG dataloader
            data_dict['verts_init_nml'] = torch.from_numpy(self.verts_init_nml[idx]).float()
            data_dict['verts_raw'] = torch.from_numpy(self.verts_raw_list[idx]).float()
            data_dict['faces_raw'] = torch.from_numpy(self.faces_raw_list[idx]).long()

        elif self.rep in ['sdf']:
            # load sdf data

            point_set_mnfld = torch.from_numpy(np.load(f"{self.sdf_dir}/{fname}.npy")).float()
            samples_nonmnfld = torch.from_numpy(np.load(f"{self.sdf_dir}/{fname}_dist_triangle.npy")).float()

            random_idx = (torch.rand(self.num_samples) * point_set_mnfld.shape[0]).long()
            point_set_mnfld = torch.index_select(point_set_mnfld, 0, random_idx)
            normal_set_mnfld = point_set_mnfld[:, 3:] 
            point_set_mnfld = point_set_mnfld[:, :3] # currently all center == [0, 0, 0], scale == 1

            random_idx = (torch.rand(self.num_samples) * samples_nonmnfld.shape[0]).long()
            samples_nonmnfld = torch.index_select(samples_nonmnfld, 0, random_idx)

            data_dict['points_mnfld'] = point_set_mnfld
            data_dict['normals_mnfld'] = normal_set_mnfld
            data_dict['samples_nonmnfld'] = samples_nonmnfld

            # load mesh data
            # raw_mesh = trimesh.load(f"{self.raw_mesh_dir}/{fname}.{self.raw_mesh_file_type}", process=False, maintain_order=True)
            # data_dict['raw_mesh_verts'] = np.asarray(raw_mesh.vertices).astype(np.float32)
            # data_dict['raw_mesh_faces'] = np.asarray(raw_mesh.faces)

        return data_dict

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                # TODO: should use torch instead of numpy here
                # if key in ['verts_raw', 'faces_raw']: # (\sum_{N_i}, d)
                #     coors = []
                #     for i, coor in enumerate(val):
                #         coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                #         coors.append(coor_pad)
                #     ret[key] = np.concatenate(coors, axis=0)
                if key in ['verts_raw', 'faces_raw']: # (B, N_max, d)
                    max_raw = max([len(x) for x in val])
                    batch_raw = torch.zeros((batch_size, max_raw, val[0].shape[-1])).float()
                    batch_raw_lengths = torch.zeros((batch_size)).long()
                    for k in range(batch_size):
                        batch_raw[k, :val[k].__len__(), :] = val[k]
                        batch_raw_lengths[k] = val[k].__len__()
                    ret[key] = batch_raw
                    ret[key + '_lengths'] = batch_raw_lengths
                else:
                    ret[key] = torch.stack(val, dim=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        # ret['batch_size'] = batch_size
        return ret


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from pyutils import *

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=str, help='sdf or mesh')
    parser.add_argument("--config", type=str, required=True, help='config yaml file path, e.g. ../config/dfaust.yaml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    update_config_from_args(config, args)

    train_dataset = DFaustDataSet(mode='train', rep=config.rep, config=config.dataset)
    test_dataset  = DFaustDataSet(mode='test',  rep=config.rep, config=config.dataset)

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    raw_mesh_list = []
    # for batch_idx, batch_dict in enumerate(test_loader):
    for batch_idx, batch_dict in enumerate(train_loader):
        for i in range(batch_size):
            if args.rep == 'sdf':
                print(i, batch_dict['points_mnfld'].shape)
                print(i, batch_dict['normals_mnfld'].shape)
            if args.rep == 'mesh':
                raise NotImplementedError

            import open3d as o3d
            import vis_utils

            starts_mnfld = batch_dict['points_mnfld'][i].numpy()
            ends_mnfld = batch_dict['points_mnfld'][i].numpy() + batch_dict['normals_mnfld'][i].numpy() * 0.1
            vf_mnfld = vis_utils.create_vector_field(starts_mnfld, ends_mnfld, [0, 1, 0])
            pcd_mnfld = vis_utils.create_pointcloud_from_points(starts_mnfld, [1, 0, 0])

            starts_nonmnfld = batch_dict['samples_nonmnfld'][i].numpy()[:, :3]
            ends_nonmnfld = batch_dict['samples_nonmnfld'][i].numpy()[:, :3] + batch_dict['samples_nonmnfld'][i].numpy()[:, 3:6] * 0.03
            vf_nonmnfld = vis_utils.create_vector_field(starts_nonmnfld, ends_nonmnfld, [0, 0, 1])
            pcd_nonmnfld = vis_utils.create_pointcloud_from_points(starts_nonmnfld, [1, 0, 0])

            raw_mesh = vis_utils.create_triangle_mesh(batch_dict['raw_mesh_verts'][i].numpy(), batch_dict['raw_mesh_faces'][i].numpy())
            raw_mesh_list.append(raw_mesh)

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            o3d.visualization.draw_geometries([raw_mesh, coord, vf_mnfld, pcd_mnfld])
            o3d.visualization.draw_geometries([coord, vf_nonmnfld, pcd_nonmnfld])
            # from IPython import embed; embed()

        break
    from IPython import embed; embed()





