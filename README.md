# GenCorres
Code for ICLR 2024 paper: [GenCorres: Consistent Shape Matching via Coupled Implicit-Explicit Shape Generative Models](https://openreview.net/pdf?id=dGH4kHFKFj).

A cleaner version of the code for training the implicit generator is in the [Supplementary Material](https://openreview.net/attachment?id=dGH4kHFKFj&name=supplementary_material).

## Dataset

We use the data processing code of [SALD](https://github.com/matanatz/SALD/tree/main). The processed dataset is in [this link](https://drive.google.com/drive/folders/1JvPRxcuqeUV9evtUNKMKgQWnhs0uzlg-?usp=drive_link).

Unzip the dataset:
```
.
├── DFAUST
│   ├── registrations
│   └── registrations_processed_sal_sigma03
└── SMAL
    ├── registrations
    └── registrations_processed_sal_sigma03
```
Change the `data_dir` in the config file (e.g. `./config/dfaust/ivae_dfaustJSM1k.yaml`).

Below we show the example of JSM for the DFAUST dataset (1k shapes). Pretrained model is in [work_dir.zip](https://drive.google.com/drive/folders/1JvPRxcuqeUV9evtUNKMKgQWnhs0uzlg-?usp=drive_link).

## Stage 1
Train the implicit network with regularization to fit the input shapes:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py --launcher pytorch --config ./config/dfaust/ivae_dfaustJSM1k.yaml --mode train --rep sdf # (--continue_from 2999)
```

To visualize the interpolation of a pair of shapes in the shape space:
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/dfaust/ivae_dfaustJSM1k.yaml --mode interp --rep sdf --continue_from 6499 --split train --interp_src_fid 50009-running_on_spot-running_on_spot.000366 --interp_tgt_fid 50002-chicken_wings-chicken_wings.004011
```

## Stage 2

### Latent space interpolation

#### Generate latents for each shape
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/dfaust/ivae_dfaustJSM1k.yaml --mode analysis --rep sdf --continue_from 6499 --split test
```
The outputs are `latents_all_test_6499.npy` and `latents_all_test_6499.pkl` in `work_dir/dfaust/ivae_dfaustJSM1k/results/test/analysis_sdf`.

#### Create KNN graph
Create a KNN graph according to the latents, also add edges from the template to all the remaining shapes.	
```
cd ./registration_dfaust
python gen_edges.py --epoch 6499 --split test --data_root ../work_dir/dfaust/ivae_dfaustJSM1k/results/test/analysis_sdf/
```
The outputs are: `work_dir/dfaust/ivae_dfaustJSM1k/results/test/analysis_sdf/edge_ids/test_6499_edge_ids_K25.npy`.

#### Interpolate shapes according to edge_ids
The command is in `interp/batch_interp.sh`. We utilize [HTCondor](https://htcondor.org/) to accelerate the execution.
```
condor_submit interp/condor.sh
```
The outputs are in: `work_dir/dfaust/ivae_dfaustJSM1k/results/test/interp_edges_sdf/6499/`. Each folder stores the interpolation results of a pair of shapes.


### Nonrigid registration between pairs

#### Prepare data for MATLAB
```
cd ./registration_dfaust
python preprocess_registration_data.py
```
The outputs are:
```
├── mesh_raw   # raw mesh
├── mesh_sim   # simplified mesh, each about 1k vertices
└── meta_test_6499_K5.mat # edge_ids and fids
```

#### Nonrigid registration
We use MATLAB to solve the following optimization problem:
```
Energy = L_point2point * beta + L_point2plane * (1 - beta) + lambda * L_arap
```

To utilize [HTCondor](https://htcondor.org/ ) to accelerate the execution:
```
cd ./registration_dfaust/multiCorres_sync_dfaust1k
condor_submit condor.sh
```
The registered meshes are in: `./dfaust1k/mesh_def`

### Propagate correspondences for initialization
```
cd ./registration_dfaust
python gen_graph.py --epoch 6499 --split test --edge_ids_path ../work_dir/dfaust/ivae_dfaustJSM1k/results/test/analysis_sdf/edge_ids/test_6499_edge_ids_K25.npy
```
The outputs are in: `./dfaust1k/mesh_corres/`

## Stage 3

### Initialize the mesh generator
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config ./config/dfaust/admesh_dfaustJSM1k.yaml --rep mesh --mode train --data_parallel
```

### Refinement
After training 999 epochs, change the hyperparameter `mesh_arap_weight` in the yaml file to `0.001` and resume training:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config ./config/dfaust/admesh_dfaustJSM1k.yaml --rep mesh --mode train --data_parallel  --continue_from 999 --batch_size 28
```

Stop training at epoch 1500. To generate the final correspondences:
```
python main.py --config ./config/dfaust/admesh_dfaustJSM1k.yaml --rep mesh --mode eval --continue_from 1499 --split train --parallel_idx 0 --parallel_interval 1000
```
The outputs are in `work_dir/dfaust/ivae_dfaustJSM1k/results/train/eval_mesh`.


## Contact
If you have any questions, you can contact Haitao Yang (yanghtr [AT] outlook [DOT] com).

