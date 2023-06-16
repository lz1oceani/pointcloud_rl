# On the Efficacy of 3D Point Cloud Reinforcement Learning

In this work, we conduct the first systematic study on the efficacy of 3D visual RL from point clouds, and compare it with well-established RL from 2D RGB/RGB-D representations. Through extensive experiments, we demonstrate that 3D point cloud representations are particularly beneficial on tasks where agent-object/object-object spatial relationship reasoning plays a crucial role, and achieves better sample complexity and performance than 2D image-based agents. Moreover, we carefully investigate the design choices for 3D point cloud RL agents from perspectives such as network inductive bias, representational robustness, data augmentation, and data post-processing. We hope that our study provides insights, inspiration, and guidance for future works on 3D visual RL.

- [On the Efficacy of 3D Point Cloud Reinforcement Learning](#on-the-efficacy-of-3d-point-cloud-reinforcement-learning)
  - [Installation](#installation)
  - [Experiment Running instructions](#experiment-running-instructions)
  - [Tips for running DM Control in headless mode](#tips-for-running-dm-control-in-headless-mode)
  - [More Repo Details](#more-repo-details)
    - [Tasks / Environments](#tasks--environments)
    - [Algorithms and Networks](#algorithms-and-networks)
    - [Config files](#config-files)
  - [Citation](#citation)
  - [License](#license)


## Installation

First, install Mujoco by downloading `https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz` (or `https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz` for MacOS). After extracting the tar file, put the `mujoco210` dirctory inside `~/.mujoco` directory (create the directory if not exist). Then, ensure that the following line is in your `~/.bashrc` file (or `~/.zshrc` if you use zsh): 
```
export LD_LIBRARY_PATH=/home/{USERNAME}/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
```

Then, install dependencies using the following procedure:

```
git clone https://github.com/lz1oceani/pointcloud_rl.git
cd pointcloud_rl
conda env create -f environment.yml
conda activate efficacy_pointcloud_rl

cd mani_skill/
pip install -e . # installs ManiSkill
pip install sapien #install sapien

cd .. # back to pointcloud_rl
pip install -e .
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt # installs DM Control and pyrl learning repo
pip uninstall torchsparse
sudo apt-get install libsparsehash-dev # brew install google-sparsehash if you use Mac OS
# before installing torchsparse, make sure the nvcc version is the same as the cuda version used when installing pytorch
pip install git+https://github.com/lz1oceani/torchsparse.git
```

For the pytorch installation, you can change the cudatoolkit version to comply with the CUDA version of your machine. 



## Experiment Running instructions 
We used random seeds 1000, 2000, 3000 in our experiments. This can be altered using the `--seed` flag.

Usage:

```
python ./pyrl/apis/run_rl.py config_file_path \
--work-dir {work_directory_path} \
[--seed {seed}] \
[--gpu-ids {gpu_ids...} | --num-gpus {number_of_gpus}] \
[--cfg-options ["env_cfg.env_name={environment_name}"] ["env_cfg.obs_mode={observation_mode}"]]
```

Explanations:

- `/pyrl/apis/run_rl.py` is the driver module that starts the training. The first argument to this module is the configuration file path. 

- Set `config_file_path` to one of the configureation files in the `configs/mfrl` directory. 

- use `--work-dir` to change the desired save location of the experiment. 

- `--seed` takes a single integer for seed. If left blank, a random seed will be used. 

- If using `--gpu-ids`, input any number of existing CUDA gpu ids. If using `--num-gpus`, input a single integer for the number of gpus to be used. To reproduce our DM Control experiments, we used `--gpu-ids 0`. To reproduce our ManiSkill experiments, we used `--gpu-ids 0 1` (or `--num-gpus 2`). 

- The `--cfg-options` is used to override part of the configuration files. It is an optional argument and is followed by any number of modifications to the configuration file. Each modification needs to follow the format `"{arg}={option}"` (*the quotation marks here are necessary*). The two most important modifications are `"env_cfg.env_name"` and `"env_cfg.obs_mode"`.

    - Replace `env_name` with one of the environment from DM control or ManiSkill. Note that the `config_file_path` need to be from the same domain of environments. For example, `"env_cfg.env_name=OpenCabinetDoor_train-v0"`.

    - Replace `obs_mode` with either `rgb`, `rgbd` or `pointcloud`. For example, `"env_cfg.obs_mode=pointcloud"`. It is not necessary to explicitly set the observation mode if the configuration file's network type is `pn` or `sparse_conv`. The `cnn` configuration files can be run under both `rgb` and `rgbd` mode, and the default is `rgb` mode.

    - You can also manually modify the `env_name` component or the `obs_mode` component of `env_cfg` inside the configuration files to accomplish the same outcome. However, using `--cfg-options` adds flexibility.  

As an example, to run DrQ with Jitter augmentation on ManiSkill's `MoveBucket_4000-v0` Environment: 

```
python ./pyrl/apis/run_rl.py ./configs/mfrl/drq/maniskill/pn_jitter.py \
--work-dir=/path/to/workdir/ --seed 1000 --num-gpus 2 --cfg-options "env_cfg.env_name=MoveBucket_4000-v0" 
```

Another example of running SAC with rgbd input on the Walker Walk task of DM Control:

```
python ./pyrl/apis/run_rl.py ./configs/mfrl/sac/dm_control/cnn.py \
--work-dir=/path/to/workdir/ --seed 1000 --gpu-ids 0 \
--cfg-options "env_cfg.env_name=dmc_walker_walk-v0" "env_cfg.obs_mode=rgbd"

# if you encounter errors when running DM Control environments in headless mode, add "MUJOCO_GL=egl" before the command.
```

For the motivating example, to run DrQ with rgbd observation mode:

```
python ./pyrl/apis/run_rl.py ./configs/mfrl/drq/dm_control/cnn_shift_motivating.py \
--work-dir=/path/to/workdir/ --seed 1000 --gpu-ids 0 --cfg-options "env_cfg.obs_mode=rgbd"

# Since there is only one environment for our motivating example, we do not need to change the environment_name. 
```


## Tips for running DM Control in headless mode

If you encounter errors when running DM Control environments in headless mode, try the following:

```
export MUJOCO_GL=egl python {command} #or MUJOCO_GL=osmesa
```


## More Repo Details

### Tasks / Environments

DM Control environment names follow this format: `dmc_{domain_name}_{task_name}-v0`. We support all DM Control environments. Specifically,

- dmc_cheetah_run-v0
- dmc_walker_walk-v0
- dmc_cartpole_swingup-v0
- dmc_reacher_easy-v0
- dmc_finger_spin-v0
- dmc_ball_in_cup_catch-v0

ManiSkill environment names follow this format: `{task_name}_[{obj_number} | train]-v0`. The environments used in our experiments are:

- Environments without object variations: 
    - OpenCabinetDoor_1000-v0
    - OpenCabinetDrawer_1000-v0
    - PushChair_3001-v0
    - MoveBucket_4000-v0
- Environments with object variations:
    - OpenCabinetDoor_train-v0
    - OpenCabinetDrawer_train-v0
    - PushChair_train-v0
    - MoveBucket_train-v0

### Algorithms and Networks
Location of algorithm implementations:

- SAC: 
`./pyrl_code_release_v1/pyrl/methods/mfrl/sac.py`
- DrQ:
`./pyrl_code_release_v1/pyrl/methods/mfrl/drq.py`

Location of network implementations: 

- CNN: `./pyrl_code_release_v1/pyrl/networks/backbones/cnn.py`
- PointNet: `./pyrl_code_release_v1/pyrl/networks/backbones/pointnet.py`
- SparseConvNet: `./pyrl_code_release_v1/pyrl/networks/backbones/sp_resnet.py`

### Config files 
Config files are located in the `configs/mfrl` directory, and are divided by algorithms (sac, drq) and environment domains (dmc or maniskill). Names of config file follow the format `network_type[_augmentation].py`. The `network_type` can be `cnn` for CNN, `pn` for PointNet, and `sparse_conv` for SparseConvNet. For DrQ configuration files, the `[augmentation]` can be `shift`, `rot`, `jitter`, `colorjitter`, or`dropout`. PointNet can use all augmentations, while CNN can only use the `shift` augmentation. 

If a configuration file contains the suffix `motivating`, they are for the motivating example we provided in our experiments and does not belong to the DM Control environments. 

For example, the configuration file for running SAC on the `walker_walk` environment with rgb or rgbd observation mode is `./configs/mfrl/sac/dm_control/cnn.py`. The configuration file for running DrQ on the MoveBucket_4000-v0 environment with point cloud observation mode and jitter augmentation is `./configs/mfrl/drq/maniskill/pn_jitter.py`.




## Citation

Please cite our paper if you find our idea helpful. Thanks a lot!

```
@article{ling2023efficacy,
  title={On the Efficacy of 3D Point Cloud Reinforcement Learning},
  author={Ling, Zhan and Yao, Yunchao and Li, Xuanlin and Su, Hao},
  journal={arXiv preprint arXiv:2306.06799},
  year={2023}
}
```

## License

This project is licensed under the Apache 2.0 license.
