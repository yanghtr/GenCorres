#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import trimesh
import argparse
import numpy as np
import scipy.io as sio
import networkx as nx

import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import point_cloud_utils as pcu

def get_template_idx(pkl, template_fid):
    for k, v in pkl.items():
        if v['fid'] == template_fid:
            template_idx = k
            return template_idx


def vis_edges_from_adj(A, pkl, mesh_root):
    import vis_utils
    # vis who can reach current shape
    num = A.shape[0]
    nbatch = 20
    for i in range((num//nbatch)):
        A_chunk = A[:, i * nbatch : (i+1) * nbatch].T # (nbatch, num)
        mesh_list = []
        for eid, edges in enumerate(A_chunk):
            src_idx = i * nbatch + eid
            interp_ids = np.where(edges)[1].tolist()
            interp_ids = [src_idx] + interp_ids
            print(i, eid, interp_ids)
            for ii, idx in enumerate(interp_ids):
                fid = pkl[idx]['fid']
                fname = '/'.join(fid.split('-')) + '.obj'
                # print(fname)
                mesh = trimesh.load(f"{mesh_root}/{fname}", process=False)
                mesh.vertices = mesh.vertices + np.array([0, 0, 2]) * ii + np.array([2, 0, 0]) * eid
                mesh_list.append(vis_utils.create_triangle_mesh(mesh.vertices, mesh.faces))
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries(mesh_list + [coord])
    from IPython import embed; embed()


def vis_edges_from_ids(vis_ids, edges_list, pkl, mesh_root):
    import vis_utils
    mesh_list = []
    for ii, idx in enumerate(vis_ids):
        fid = pkl[idx]['fid']
        fname = '/'.join(fid.split('-')) + '.obj'
        mesh = trimesh.load(f"{mesh_root}/{fname}", process=False)
        mesh.vertices = mesh.vertices + np.array([2, 0, 0]) * ii
        mesh_list.append(vis_utils.create_triangle_mesh(mesh.vertices, mesh.faces))
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries(mesh_list + [coord])


if __name__ == '__main__':
    '''
        Example commands: python gen_graph.py --epoch 6499 --split test
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True, default=None, help='e.g. 2999 or 6499')
    parser.add_argument("--split", type=str, required=True, default='test', help='{train, test}, use train or test dataset')
    parser.add_argument("--analysis_dir", type=str,\
                        default='../work_dir/dfaust/dfaust1kBak/ivae_dfaustTest1k_8B8_lr1k_arap_8B8_SE3k_inv_SLS_w1e-3/results/test/analysis_sdf/',\
                        help='dir to e.g. latents_all_test_5499.npy and latents_all_test_5499.pkl')
    parser.add_argument("--interp_dir", type=str, default='./dfaust1k/mesh_def/', help='dir to interpolated meshes')
    parser.add_argument("--dump_dir", type=str, default='./dfaust1k/mesh_corres/', help='dir to mesh init and correspondence')
    parser.add_argument("--data_dir", type=str, default='/scratch/cluster/yanght/Dataset/Human/DFAUST/registrations/', help='dir to interpolated meshes')
    parser.add_argument("--edge_ids_path", type=str, required=True, default='{args.analysis_dir}/edge_ids/{args.split}_{args.epoch}_edge_ids_K5.npy', help='path to edge_ids npy')
    args = parser.parse_args()

    assert(args.split == 'test')

    template_fid = '50009-running_on_spot-running_on_spot.000366'

    pkl_path = f'{args.analysis_dir}/latents_all_{args.split}_{args.epoch}.pkl'
    npy_path = f'{args.analysis_dir}/latents_all_{args.split}_{args.epoch}.npy'
    # edge_ids_path = f'{args.analysis_dir}/edge_ids/{args.split}_{args.epoch}_edge_ids_K5.npy'
    pkl = pickle.load(open(pkl_path, 'rb'))
    edge_ids = np.load(args.edge_ids_path)

    template_idx = get_template_idx(pkl, template_fid)
    assert(template_idx == 374)
    num_nodes = len(pkl)
    assert(num_nodes == 1000)
    template_fname = '/'.join(template_fid.split('-')) + '.obj'
    mesh_template = trimesh.load(f"{args.data_dir}/{template_fname}", process=False, maintain_order=True)

    corres_dict = {}
    dists_dict = {}
    for eid, (sid, tid) in enumerate(edge_ids):
        sfid = pkl[sid]['fid']
        tfid = pkl[tid]['fid']
        mesh_def = trimesh.load(f"{args.interp_dir}/meshdef_{sid}_{tid}.obj", process=False, maintain_order=True)

        sfname = '/'.join(sfid.split('-')) + '.obj'
        mesh_src = trimesh.load(f"{args.data_dir}/{sfname}", process=False, maintain_order=True)
        tfname = '/'.join(tfid.split('-')) + '.obj'
        mesh_tgt = trimesh.load(f"{args.data_dir}/{tfname}", process=False, maintain_order=True)

        dists_def_to_tgt, corres_def_to_tgt = pcu.k_nearest_neighbors(mesh_def.vertices, mesh_tgt.vertices, k=1)
        dists_tgt_to_def, corres_tgt_to_def = pcu.k_nearest_neighbors(mesh_tgt.vertices, mesh_def.vertices, k=1)
        # corres = corres_tgt_to_def[corres_def_to_tgt]
        # diff = np.linalg.norm(mesh_template.vertices[corres] - mesh_template.vertices, axis=-1)
        dists_dict[(sid, tid)] = dists_tgt_to_def
        corres_dict[(sid, tid)] = corres_def_to_tgt

        print(f"eid={eid}, sid={sid}, tid={tid}, sfid={sfid}, tfid={tfid}")

    dists_mean_dict = {}
    for eid, (sid, tid) in enumerate(edge_ids):
        dists_mean_dict[(sid, tid)] = dists_dict[(sid, tid)].mean()
    dists_min = np.min([v for k, v in dists_mean_dict.items()])

    weights_dict = {}
    # for eid, (sid, tid) in enumerate(edge_ids):
    #     w_diff = dists_dict[(sid, tid)].mean() - dists_min
    #     weights_dict[(sid, tid)] = w_diff * w_diff
    # IMPORTANT NOTE: instead of using mean, use distribution to filter out bad edges
    for eid, (sid, tid) in enumerate(edge_ids):
        w_diff = dists_dict[(sid, tid)].copy()
        w_diff.sort()
        weights_dict[(sid, tid)] = w_diff.mean() if w_diff[6810] < 0.02 else 100

    G = nx.DiGraph()
    G.add_nodes_from(np.arange(num_nodes)) # use indices as labels
    G.add_edges_from(edge_ids)
    nx.set_edge_attributes(G, values = weights_dict, name = 'weight')

    lengths, paths = nx.single_source_dijkstra(G, template_idx)

    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)
    for mesh_idx in range(num_nodes):
        print(mesh_idx)
        path = paths[mesh_idx]
        assert(path[0] == template_idx)

        corres = np.arange(mesh_template.vertices.shape[0])
        if mesh_idx != template_idx:
            for ii in range(len(path[:-1])):
                sid = path[ii]
                tid = path[ii + 1]
                corres = corres_dict[(sid, tid)][corres]

        fid = pkl[mesh_idx]['fid']
        fname = '/'.join(fid.split('-')) + '.obj'
        mesh = trimesh.load(f"{args.data_dir}/{fname}", process=False, maintain_order=True)
        verts_corres = mesh.vertices[corres]
        faces_corres = mesh_template.faces
        mesh_corres = trimesh.Trimesh(vertices=verts_corres, faces=faces_corres, process=False)

        mesh_corres.export(f"{args.dump_dir}/{fid}_init.obj")
        np.save(f"{args.dump_dir}/{fid}_corres.npy", corres)







