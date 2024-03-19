import os
import sys
import pickle
import trimesh
import argparse
import numpy as np
import scipy.io as sio


def get_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


if __name__ == '__main__':
    
    ##### convert python file to matlab file
    template_fid = '50009-running_on_spot-running_on_spot.000366'
    analysis_root = '../work_dir/dfaust/ivae_dfaustJSM1k/results/test/analysis_sdf/'
    dump_root = './dfaust1k/'
    epoch = 6499
    num_neighbors = 25 # 25

    edge_ids = np.load(f'{analysis_root}/edge_ids/test_{epoch}_edge_ids_K{num_neighbors}.npy')
    
    pkl = pickle.load(open(f"{analysis_root}/latents_all_test_{epoch}.pkl", 'rb'))
    N = len(pkl)
    fids = [pkl[i]['fid'] for i in range(N)]
    template_idx = fids.index(template_fid)
    assert(template_idx == 374)

    metadata = {
        'edge_ids': edge_ids,
        'fids': fids,
        'template_idx': template_idx,
        'template_fid': template_fid,
    }
    sio.savemat(f'{dump_root}/meta_test_{epoch}_K{num_neighbors}.mat', mdict=metadata)

    ##### aggregate dataset
    data_root = '/scratch/cluster/yanght/Dataset/Human/DFAUST/registrations/'
    mesh_raw_dir = get_directory( f'{dump_root}/mesh_raw/' )
    mesh_sim_dir = get_directory( f'{dump_root}/mesh_sim/' )

    for i, fid in enumerate(fids):
        print(i, fid)
        fname = '/'.join(fid.split('-')) + '.obj'
        fpath = f'{data_root}/{fname}'
        os.system(f"cp {fpath} {mesh_raw_dir}/{fid}.obj")

        mesh = trimesh.load(fpath, process=False, maintain_order=True)
        mesh_sim = mesh.simplify_quadratic_decimation(2000)
        mesh_sim.export(f"{mesh_sim_dir}/{fid}_sim.obj")



