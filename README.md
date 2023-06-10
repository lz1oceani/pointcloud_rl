# Installation
Start with installing ManiSkill first using the following procedure (from the parent directory of this README file):
```
git clone https://github.com/xuanlinli17/corl_22_frame_mining.git
cd corl_22_frame_mining/
git checkout dev
cd maniskill/
conda env create -f environment.yml
conda activate mani_skill
pip install -e .
pip install sapien #install sapien
```
You will also need to install Mujoco by first download `https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz` (or `https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz` for MacOS). After extract the tar file, put the `mujoco210` dirctory inside `~/.mujoco` directory (create the directory if not existing). Lastly you need to run 
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin 
```

After installing ManiSkill and dependencies to DM Control, run the following inside the conda environment mani_skill: 
```
cd ../pyrl/
pip install -e .
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt 
pip uninstall torchsparse
sudo apt-get install libsparsehash-dev # brew install google-sparsehash if you use Mac OS
# before installing torchsparse, make sure the nvcc version is the same as the cuda version used when installing pytorch
pip install git+https://github.com/lz1oceani/torchsparse.git
```

For the pytorch installation, you can change the cudatoolkit virsion to comply with the CUDA version of your machine. 

# Tasks

The DM Control environments are named in the following way: `dmc_{domain_name}_{task_name}-v0`.

All DM Control environments are supported. In our experiments, we used:

- dmc_cheetah_run-v0
- dmc_walker_walk-v0
- dmc_cartpole_swingup-v0
- dmc_reacher_easy-v0
- dmc_finger_spin-v0
- dmc_ball_in_cup_catch-v0

The ManiSkil environments are named as `{task_name}_[{obj_number} | train]-v0`

The ManiSkill environment we used in our experiment are:

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

# Algorithms and network locations
Location of algorithm implementations:

- SAC: 
`./pyrl_code_release_v1/pyrl/methods/mfrl/sac.py`
- DrQ:
`./pyrl_code_release_v1/pyrl/methods/mfrl/drq.py`

Location of network implementations: 

- CNN: `./pyrl_code_release_v1/pyrl/networks/backbones/cnn.py`
- PointNet: `./pyrl_code_release_v1/pyrl/networks/backbones/pointnet.py`
- Sparse ConvNet: `./pyrl_code_release_v1/pyrl/networks/backbones/sp_resnet.py`

# Config files 
Config files are located in the `configs/mfrl` directory, and are divided by algorithms (sac, drq) and environment domains (dmc or maniskill). The config file names follows the format `network_type[_augmentation].py`. The `network_type` can be `cnn` for CNN, `pn` for PointNet, and `sparse_conv` for Sparse Conv nets. For configuration files of DrQ, the `[augmentation]` can be `shift`, `rot`,`jitter`,`colorjitter`,or`droptout`. Only the `shift` augmentation can be used for `cnn`. 

If a configuration file contains the suffix `motivating`, they are for the motivating example we provided in our experiments and does not belong to the DM Control environments. 

For example, the configuration file to run the SAC algorithm on the walker_walk environment in rgb or rgbd mode is: 

`./configs/mfrl/sac/dm_control/cnn.py`

Another example of the configuration file to run the DrQ algorithm on the MoveBucket_4000-v0 enviroment in point cloud mode using jitter augmentation:

`./configs/mfrl/drq/maniskill/pn_jitter.py` 

# Run instructions 
We used random seeds 1000, 2000, 3000 in our experiments. This can be altered using the `--seed` flag.

Usage:

```
python ./pyrl/apis/run_rl.py config_file_path --work-dir {work_directory_path} [--seed {seed}] [--gpu-ids {gpu_ids...} | --num-gpus {number_of_gpus}] [--cfg-options ["env_cfg.env_name={environment_name}"] ["env_cfg.obs_mode={observation_mode}"]]
```

- `/pyrl/apis/run_rl.py` is the driver module that starts the training. The first argument to this module is the configuration file path. 

- Set `config_file_path` to one of the configureation files in the `configs/mfrl` directory. 

- use `--work-dir` to change the desired save location of the experiment. 

- `--seed` takes a single integer for seed. If left blank, a random seed will be used. 

- If using `--gpu-ids`, input any number of existing CUDA gpu ids. If using `--num-gpus`, input a single integer for the number of gpus to be used. To reproduce our DM Control experiments, we used `--gpu-ids 0`. To reproduce our ManiSkill experiments, we used `--num-gpus 2`. 

- The `--cfg-options` is used to override part of the configuration files. It is an optional argument and is followed by any number of modifications to the configuration file. The modifications need to be within quotation marks. The configuration files are structured using python dictionaries, and the modificaiton strings follows the exact format of modifying a python dictionary using the `=` assign operator. The two most imporant modifications are `"env_cfg.env_name"` and `"env_cfg.obs_mode"`.

    - Replace `environment_name` with one of the enviornment from DM control or ManiSkill. Note that the `config_file_path` need to be from the same domain of environments. The quot marks are necessary. 

    - Replace `observation_mode` to either `rgb`, `rgbd` or `pointcloud`. It is not necessary to explicitly set the observation mode if the configuration file's network type is `pn` or `sparse_conv`. The `cnn` configuration files can be run under both `rgb` and `rgbd` mode, and the default is `rgb` mode. The quot marks are necessary. 

    - You can also directly modify the `env_name` component or `obs_mode` component of `env_cfg` manually inside the configuration file to achieve the same result. However, using `--cfg-options` adds flexibility.  

Example of running DrQ algorithm with Jitter augmentation on ManiSkill's MoveBucket_4000-v0 Environment: 

```
python ./pyrl/apis/run_rl.py ./configs/mfrl/drq/maniskill/pn_jitter.py --work-dir=/path/to/workdir/ --seed 1000 --num-gpus 2 --cfg-options "env_cfg.env_name=MoveBucket_4000-v0" 
```

Another example of running SAC algorithm with rgbd input on the Walker Walk task of DM Control:

```
python ./pyrl/apis/run_rl.py ./configs/mfrl/sac/dm_control/cnn.py --work-dir=/path/to/workdir/ --seed 1000 --gpu-ids 0 --cfg-options "env_cfg.env_name=dmc_walker_walk-v0" "env_cfg.obs_mode=rgbd"
```

To run the motivating example we provided in our experiments, for example, using DrQ algorithm on rgbd observation mode, run:

```
python ./pyrl/apis/run_rl.py ./configs/mfrl/drq/dm_control/cnn_shift_motivating.py --work-dir=/path/to/workdir/ --seed 1000 --gpu-ids 0 --cfg-options "env_cfg.obs_mode=rgbd"
```

Since there is only one environment for motivating examples, we do not need to change the environment_name. 

## Tips for running DM Control in headless mode
Before running DM Control environments, set the MUJOCO_GL environment variable by 
```
export MUJOCO_GL=egl #or MUJOCO_GL=osmesa
```
to run in headless mode. 
