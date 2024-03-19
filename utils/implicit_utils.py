import os
import time
import torch
import numpy as np
from skimage import measure
from tqdm import tqdm
from .diff_operators import gradient


def sdf_decode_mesh_from_single_lat(model, latent_vec, resolution=256, voxel_size=None, max_batch=int(2 ** 18), offset=None, scale=None, points_for_bound=None, verbose=False, x_range=[-1, 1], y_range=[-0.7, 1.7], z_range=[-1.1, 0.9]):
    '''
    Args:
        model: only model.decoder is used
        resolution: the resolution of the shortest_axis
        latent_vec: (d, )
    '''
    if resolution is not None:
        assert(voxel_size is None)
        if points_for_bound is not None:
            grid = get_grid_YXZ(points_for_bound, resolution)
        else:
            grid = get_grid_uniform_YXZ(resolution, x_range=x_range, y_range=y_range, z_range=z_range)
    else:
        assert(voxel_size is not None)
        grid = get_grid_from_size_YXZ(points_for_bound, voxel_size)

    sdf_volume_yxz = []
    ptn_samples_list = torch.split(grid['grid_points'], max_batch, dim=0)
    if verbose:
        ptn_samples_list = tqdm(ptn_samples_list)
    for ptn_samples in ptn_samples_list:
        ptn_samples.requires_grad = False
        sdf_samples = model.decoder(ptn_samples[None, :, :].to(latent_vec.device), latent_vec[None, :])
        sdf_samples = sdf_samples[0, :, 0].detach().cpu().numpy()
        sdf_volume_yxz.append(sdf_samples)

    sdf_volume = np.concatenate(sdf_volume_yxz, axis=0).reshape(grid['ysize'], grid['xsize'], grid['zsize']).transpose([1, 0, 2]) # XYZ

    assert(np.min(sdf_volume) < 0 and np.max(sdf_volume) > 0)

    verts, faces = convert_sdf_volume_to_verts_faces(sdf_volume, grid['voxel_grid_origin'], grid['voxel_size'])

    return verts, faces


def convert_sdf_volume_to_verts_faces(sdf_volume, voxel_grid_origin, voxel_size, offset=None, scale=None):
    """
    Args:
        sdf_volume: (X, Y, Z)
        voxel_grid_origin: (3,) bottom, left, down origin of the voxel grid
        voxel_size: float, the size of the voxels
    """

    verts, faces, normals, values = measure.marching_cubes(
        volume=sdf_volume, level=0.0, spacing=[voxel_size] * 3
    )
    verts = verts + voxel_grid_origin

    # apply additional offset and scale. based on preprocess_dfaust.py
    if scale is not None:
        mesh_points = mesh_points * scale
    if offset is not None:
        mesh_points = mesh_points + offset

    return verts, faces


def get_grid_YXZ(points, resolution):
    ''' For x, y, z, the voxel sizes are the same but grid sizes are different
    Args:
        points: (S, 3)
        resolution: the resolution of the shortest_axis
    Returns:
        grid_points: (Ny * Nx * Nz, 3), the order is Y, X, Z instead of X, Y, Z
    '''
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().detach().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().detach().cpu().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        voxel_size = length / (x.shape[0] - 1)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        voxel_size = length / (y.shape[0] - 1)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        voxel_size = length / (z.shape[0] - 1)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z) # default: indexing='xy', return shape is (N2, N1, N3) instead of (N1, N2, N3)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    xsize, ysize, zsize = x.shape[0], y.shape[0], z.shape[0]
    voxel_grid_origin = np.array([x[0], y[0], z[0]])

    return {"grid_points":grid_points,
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "xsize": xsize,
            "ysize": ysize,
            "zsize": zsize,
            "voxel_grid_origin": voxel_grid_origin,
            "voxel_size": voxel_size,
            "shortest_axis_index":shortest_axis}


def get_grid_uniform_YXZ(resolution, x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2]):
    ''' For x, y, z, the voxel sizes are the same but grid sizes are different
    Args:
        resolution: the resolution of the shortest_axis, i.e. x axis
    Returns:
        grid_points: (Ny * Nx * Nz, 3), the order is Y, X, Z instead of X, Y, Z
    '''
    range_len_list = [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]
    shortest_axis = np.argmin(range_len_list)

    if shortest_axis == 0:
        x = np.linspace(x_range[0], x_range[1], resolution)
        shortest_axis_length = x.max() - x.min()
        voxel_size = shortest_axis_length / (x.shape[0] - 1)
        y = np.arange(y_range[0], y_range[1] + voxel_size, voxel_size)
        z = np.arange(z_range[0], z_range[1] + voxel_size, voxel_size)
    elif shortest_axis == 1:
        y = np.linspace(y_range[0], y_range[1], resolution)
        shortest_axis_length = y.max() - y.min()
        voxel_size = shortest_axis_length / (y.shape[0] - 1)
        x = np.arange(x_range[0], x_range[1] + voxel_size, voxel_size)
        z = np.arange(z_range[0], z_range[1] + voxel_size, voxel_size)
    elif shortest_axis == 2:
        z = np.linspace(z_range[0], z_range[1], resolution)
        shortest_axis_length = z.max() - z.min()
        voxel_size = shortest_axis_length / (z.shape[0] - 1)
        x = np.arange(x_range[0], x_range[1] + voxel_size, voxel_size)
        y = np.arange(y_range[0], y_range[1] + voxel_size, voxel_size)
    else:
        raise NotImplementedError

    xx, yy, zz = np.meshgrid(x, y, z) # default: indexing='xy', return shape is (N2, N1, N3) instead of (N1, N2, N3)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    xsize, ysize, zsize = x.shape[0], y.shape[0], z.shape[0]
    voxel_grid_origin = np.array([x[0], y[0], z[0]])

    return {"grid_points": grid_points,
            "shortest_axis_length": shortest_axis_length,
            "xyz": [x, y, z],
            "xsize": xsize,
            "ysize": ysize,
            "zsize": zsize,
            "voxel_grid_origin": voxel_grid_origin,
            "voxel_size": voxel_size,
            "shortest_axis_index": shortest_axis}


def get_grid_from_size_YXZ(points, voxel_size):
    ''' For x, y, z, the voxel sizes are the same but grid sizes are different
    Args:
        points: (S, 3)
        voxel_size: float
    Returns:
        grid_points: (Ny * Nx * Nz, 3), the order is Y, X, Z instead of X, Y, Z
    '''
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().detach().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().detach().cpu().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)

    x = np.arange(input_min[0] - eps, input_max[0] + voxel_size + eps, voxel_size)
    y = np.arange(input_min[1] - eps, input_max[1] + voxel_size + eps, voxel_size)
    z = np.arange(input_min[2] - eps, input_max[2] + voxel_size + eps, voxel_size)

    if (shortest_axis == 0):
        length = np.max(x) - np.min(x)
    elif (shortest_axis == 1):
        length = np.max(y) - np.min(y)
    elif (shortest_axis == 2):
        length = np.max(z) - np.min(z)

    xx, yy, zz = np.meshgrid(x, y, z) # default: indexing='xy', return shape is (N2, N1, N3) instead of (N1, N2, N3)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    xsize, ysize, zsize = x.shape[0], y.shape[0], z.shape[0]
    voxel_grid_origin = np.array([x[0], y[0], z[0]])

    return {"grid_points":grid_points,
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "xsize": xsize,
            "ysize": ysize,
            "zsize": zsize,
            "voxel_grid_origin": voxel_grid_origin,
            "voxel_size": voxel_size,
            "shortest_axis_index":shortest_axis}



def langevin_dynamics(model, latent_vec, xyz, num_iters=1):
    '''
    Args:
        model: only model.decoder is used
        latent_vec: (d, )
        xyz: (N, 3)
    '''
    device = latent_vec.device
    assert(len(latent_vec.shape) == 1 and len(xyz.shape) == 2 and xyz.shape[1] == 3)

    xyz = torch.from_numpy(xyz).float().to(device)
    latents = latent_vec[None, :].repeat(xyz.shape[0], 1)
    
    for _ in range(num_iters):
        xyz = xyz.clone().detach().requires_grad_(True)
        sdf = model.decoder(xyz, latents) # (N, 1)
        fxyz = gradient(sdf, xyz) # (N, 3)
        xyz = xyz - sdf * fxyz

    return xyz












