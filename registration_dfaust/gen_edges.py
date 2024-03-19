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
        Example commands: python gen_edges.py --epoch 5499 --split test
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True, default=None, help='e.g. 2999 or 5499')
    parser.add_argument("--split", type=str, required=True, default='test', help='{train, test}, use train or test dataset')
    parser.add_argument("--data_root", type=str,\
                        default='./data/dfaust1k/arapTest1k/ivae_dfaustTest1k_8B8_lr1k_arap_8B8_SE3k_inv_SLS_w1e-3/results/test/analysis_sdf/',\
                        help='dir to e.g. latents_all_test_5499.npy and latents_all_test_5499.pkl')
    args = parser.parse_args()

    assert(args.split == 'test')

    template_fid = '50009-running_on_spot-running_on_spot.000366'

    pkl_path = f'{args.data_root}/latents_all_{args.split}_{args.epoch}.pkl'
    npy_path = f'{args.data_root}/latents_all_{args.split}_{args.epoch}.npy'
    pkl = pickle.load(open(pkl_path, 'rb'))

    template_idx = get_template_idx(pkl, template_fid)

    latents_all = np.array([v['latent'] for k, v in pkl.items()])
    latents_all_tmp = np.load(npy_path)
    assert(np.all(latents_all_tmp == latents_all))

    # fit neigh
    num_neighbors = 25 # 5: 6193 edges, 10: 12617 edges
    neigh = NearestNeighbors(n_neighbors=num_neighbors)
    neigh.fit(latents_all)

    _, neigh_ids = neigh.kneighbors(latents_all)
    assert(np.all(neigh_ids[:, 0] == np.arange(latents_all.shape[0])))
    src_ids = np.tile(neigh_ids[:, 0:1], (1, num_neighbors - 1)) # (N, num_neighbors - 1)
    tgt_ids = neigh_ids[:, 1:] # (N, num_neighbors - 1)

    edge_ids = np.stack((src_ids, tgt_ids), axis=-1).reshape(-1, 2)
    edge_ids = np.concatenate((edge_ids, edge_ids[:, [1, 0]]), axis=0) # (E, 2)

    # template to all
    template_ids = np.array([template_idx] * latents_all.shape[0])
    edge_template_ids = np.concatenate((template_ids[:, None], neigh_ids[:, 0:1]), axis=-1) # (N, 2)

    # template KNN
    # num_neighbors_temp = 500
    # _, neigh_ids_temp = neigh.kneighbors(latents_all[template_idx, :][None, :], n_neighbors=num_neighbors_temp)
    # src_temp_ids = np.array([template_idx] * (num_neighbors_temp - 1))
    # tgt_temp_ids = neigh_ids_temp[0, 1:]
    # edge_template_ids = np.stack((src_temp_ids, tgt_temp_ids), axis=-1).reshape(-1, 2)

    edge_ids = np.concatenate((edge_ids, edge_template_ids), axis=0) # (E+N, 2)

    G = nx.DiGraph()
    G.add_nodes_from(neigh_ids[:, 0]) # use indices as labels
    G.add_edges_from(edge_ids)
    G.remove_edges_from(nx.selfloop_edges(G)) # the only self loop is template to template

    edge_ids_new = np.array(G.edges)
    print(f"\n num of edges: {edge_ids_new.shape[0]} \n")
    dump_root = f"{args.data_root}/edge_ids/"
    if not os.path.exists(dump_root):
        os.makedirs(dump_root)
    np.save(f"{dump_root}/{args.split}_{args.epoch}_edge_ids_K{num_neighbors}.npy", edge_ids_new)
    # np.save(f"{dump_root}/{args.split}_{args.epoch}_edge_ids_K{num_neighbors}_tempKNN.npy", edge_ids_new)

    ############################## visialuzation ##############################
    # from IPython import embed; embed()
    # mesh_root = '/media/yanghaitao/HaitaoYang/Graphicsai_Backup/mnt/yanghaitao/Dataset/DFAUST/registrations/'
    # for each mesh, visualize: from which mesh we can reach the current mesh with 1 step
    # A = nx.adjacency_matrix(G).todense()
    # vis_edges_from_adj(A, pkl, mesh_root)

    # draw G
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()

    # draw specific shapes
    # vis_ids = [249, 805, 808]
    # vis_edges_from_ids(vis_ids, edge_ids, pkl, mesh_root)

    # from IPython import embed; embed()






