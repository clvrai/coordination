# Learning to Coordinate Manipulation Skills via Skill Behavior Diversification

[Youngwoon Lee](https://youngwoon.github.io), [Jingyun Yang](https://yjy0625.github.io), [Joseph J. Lim](https://clvrai.com) at [USC CLVR lab](https://clvrai.com)<br/>
[[Project website](https://clvrai.com/coordination)] [[Paper](https://openreview.net/forum?id=ryxB2lBtvH)]


This project is a PyTorch implementation of [Learning to Coordinate Manipulation Skills via Skill Behavior Diversification](https://clvrai.com/coordination), which is published in ICLR 2020.

<p align="center">
    <img src="docs/img/teaser.gif">
</p>

When mastering a complex manipulation task, humans often decompose the task into sub-skills of their body parts, practice the sub-skills independently, and then execute the sub-skills together. Similarly, a robot with multiple end-effectors can perform a complex task by coordinating sub-skills of each end-effector. To realize temporal and behavioral coordination of skills, we propose a modular framework that ﬁrst individually trains sub-skills of each end-effector with skill behavior diversiﬁcation, and learns to coordinate end-effectors using diverse behaviors of the skills.


## Prerequisites
- Ubuntu 18.04
- Python 3.6
- Mujoco 2.0


## Installation
1. Install Mujoco 2.0 and add the following environment variables into `~/.bashrc` or `~/.zshrc`

```bash
# Download mujoco 2.0
$ wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip
$ unzip mujoco.zip -d ~/.mujoco
$ mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200

# Copy mujoco license key `mjkey.txt` to `~/.mujoco`

# Add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

# For GPU rendering (replace 418 with your nvidia driver version)
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418

# Only for a headless server
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-418/libGL.so
```

2. Install python dependencies
```bash
# Run the next line for installing packages
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev python3-pip python3-numpy python3-scipy

# Install python packages
$ pip3 install -r requirements.txt

# Add wandb API key (get one from https://wandb.com)
$ wandb login WANDB_API_KEY
```

### Mujoco GPU rendering
To use GPU rendering for mujoco, you need to add `/usr/lib/nvidia-000` (`000` should be replaced with your NVIDIA driver version) to `LD_LIBRARY_PATH` before installing `mujoco-py`.
Then, during `mujoco-py` compilation, it will show you `linuxgpuextension` instead of `linuxcpuextension`.
In Ubuntu 18.04, you may encounter an GL-related error while building `mujoco-py`, open `venv/lib/python3.6/site-packages/mujoco_py/gl/eglshim.c` and comment line 5 `#include <GL/gl.h>` and line 7 `#include <GL/glext.h>`.

### Virtual display on headless machines
On servers, you don’t have a monitor. Use this to get a virtual monitor for rendering and put `DISPLAY=:1` in front of a command.
```bash
# Run the next line for Ubuntu
$ sudo apt-get install xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev

# Configure nvidia-x
$ sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# Launch a virtual display
$ sudo /usr/bin/X :1 &

# Run a command with DISPLAY=:1
DISPLAY=:1 <command>
```


## How to run experiments

0. Launch a virtual display (only for a headless server)
```bash
sudo /usr/bin/X :1 &
```

1. Train primitives

  - Jaco Pick-Push-Place and Jaco Bar-Moving

```
# With Skill Behavior Diversification (SBD)
python3 -m rl.main --env two-jaco-pick-v0 --gpu 0 --prefix right-d --seed 1 --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False --discriminator_loss_weight 10 --save_qpos True --max_global_step 300000
python3 -m rl.main --env two-jaco-pick-v0 --gpu 0 --prefix left-d --seed 1 --subdiv left_arm,left_hand,cube1-left_arm --env_args train_right-False/train_left-True --discriminator_loss_weight 10 --save_qpos True --max_global_step 300000
python3 -m rl.main --env two-jaco-place-v0 --gpu 0 --prefix right-center-d --seed 1 --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False --discriminator_loss_weight 10 --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.right-d.1/video --max_global_step 300000
python3 -m rl.main --env two-jaco-place-v0 --gpu 0 --prefix right-side-d --seed 1 --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False/dest_center-False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.right-d.1/video --max_global_step 300000
python3 -m rl.main --env two-jaco-place-v0 --gpu 0 --prefix left-side-d --seed 1 --subdiv left_arm,left_hand,cube1-left_arm --env_args train_right-False/train_left-True/dest_center-False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.left-d.1/video --max_global_step 300000
python3 -m rl.main --env two-jaco-push-v0 --gpu 0 --prefix left-d --seed 1 --subdiv left_arm,left_hand,cube1-left_arm --env_args train_left-True/train_right-False --discriminator_loss_weight 10 --save_qpos True --max_global_step 300000

# Without SBD
python3 -m rl.main --env two-jaco-pick-v0 --gpu 0 --prefix right-n --seed 1 --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False --diayn False --save_qpos True --max_global_step 300000
python3 -m rl.main --env two-jaco-pick-v0 --gpu 0 --prefix left-n --seed 1 --subdiv left_arm,left_hand,cube1-left_arm --env_args train_right-False/train_left-True --diayn False --save_qpos True --max_global_step 300000
python3 -m rl.main --env two-jaco-place-v0 --gpu 0 --prefix right-center-n --seed 1 --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False --diayn False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.right-n.1/video --max_global_step 300000
python3 -m rl.main --env two-jaco-place-v0 --gpu 0 --prefix right-side-n --seed 1 --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False/dest_center-False --diayn False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.right-n.1/video --max_global_step 300000
python3 -m rl.main --env two-jaco-place-v0 --gpu 0 --prefix left-side-n --seed 1 --subdiv left_arm,left_hand,cube1-left_arm --env_args train_right-False/train_left-True/dest_center-False --diayn False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.left-n.1/video --max_global_step 300000
python3 -m rl.main --env two-jaco-push-v0 --gpu 0 --prefix left-n --seed 1 --subdiv left_arm,left_hand,cube1-left_arm --env_args train_left-True/train_right-False --diayn False --save_qpos True --max_global_step 300000
```

  - Ant Push
```bash
# With SBD
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-down-d --env_args ant-1/direction-down --seed 1 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-up-d --env_args ant-1/direction-up --seed 1 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-right-d --env_args ant-1/direction-right --seed 1 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-left-d --env_args ant-1/direction-left --seed 1 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-down-d --env_args ant-2/direction-down --seed 1 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-up-d --env_args ant-2/direction-up --seed 1 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-right-d --env_args ant-2/direction-right --seed 1 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-left-d --env_args ant-2/direction-left --seed 1 --max_global_step 500000

# SBD with different diversity coefficient
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-down-d-005 --env_args ant-1/direction-down --seed 1 --env_args diayn_reward-0.05 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-up-d-005 --env_args ant-1/direction-up --seed 1 --env_args diayn_reward-0.05 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-right-d-005 --env_args ant-1/direction-right --seed 1 --env_args diayn_reward-0.05 --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-left-d-005 --env_args ant-1/direction-left --seed 1 --env_args diayn_reward-0.05 --max_global_step 500000

# Without SBD
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-down-n --env_args ant-1/direction-down --seed 1 --diayn False --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-up-n --env_args ant-1/direction-up --seed 1 --diayn False --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-right-n --env_args ant-1/direction-right --seed 1 --diayn False --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-1-left-n --env_args ant-1/direction-left --seed 1 --diayn False --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-down-n --env_args ant-2/direction-down --seed 1 --diayn False --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-up-n --env_args ant-2/direction-up --seed 1 --diayn False --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-right-n --env_args ant-2/direction-right --seed 1 --diayn False --max_global_step 500000
python3 -m rl.main --env ant-forward-v0 --gpu 0 --prefix ant-2-left-n --env_args ant-2/direction-left --seed 1 --diayn False --max_global_step 500000
```

2. Train composite skills

  - Jaco Bar-Moving
```
# RL baseline
python3 -m rl.main --env two-jaco-bar-moving-v0 --gpu 0 --prefix RL --seed 1 --meta hard --meta_update_target LL --subdiv left_arm,left_hand,right_arm,right_hand,cube1,cube2-left_arm,right_arm --diayn False

# RL with SBD
python3 -m rl.main --env two-jaco-bar-moving-v0 --gpu 0 --prefix RL-SBD --seed 1 --meta hard --meta_update_target both --subdiv left_arm,left_hand,right_arm,right_hand,cube1,cube2-left_arm,right_arm --max_meta_len 1

# MARL baseline
python3 -m rl.main --env two-jaco-bar-moving-v0 --gpu 0 --prefix MARL --seed 1 --meta hard --meta_update_target LL --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --diayn False

# MARL with SBD
python3 -m rl.main --env two-jaco-bar-moving-v0 --gpu 0 --prefix MARL-SBD --seed 1 --meta hard --meta_update_target both --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --max_meta_len 1

# Modular without SBD
python3 -m rl.main --env two-jaco-bar-moving-v0 --gpu 0 --prefix Modular --seed 1 --meta hard --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --subdiv_skills rl.two-jaco-pick-v0.left-n.1,rl.two-jaco-place-v0.left-side-n.1/rl.two-jaco-pick-v0.right-n.1,rl.two-jaco-place-v0.right-side-n.1 --max_meta_len 1 --diayn False

# Modular with SBD (Ours)
python3 -m rl.main --env two-jaco-bar-moving-v0 --gpu 0 --prefix ours --seed 1 --meta hard --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --subdiv_skills rl.two-jaco-pick-v0.left-d.1,rl.two-jaco-place-v0.left-side-d.1/rl.two-jaco-pick-v0.right-d.1,rl.two-jaco-place-v0.right-side-d.1 --max_meta_len 1
```

  - Jaco Pick-Push-Place
```
# RL baseline
python3 -m rl.main --env two-jaco-pick-push-place-v0 --gpu 0 --prefix RL --seed 1 --meta hard --meta_update_target LL --subdiv left_arm,left_hand,right_arm,right_hand,cube1,cube2-left_arm,right_arm --diayn False

# RL with SBD
python3 -m rl.main --env two-jaco-pick-push-place-v0 --gpu 0 --prefix RL-SBD --seed 1 --meta hard --meta_update_target both --subdiv left_arm,left_hand,right_arm,right_hand,cube1,cube2-left_arm,right_arm --max_meta_len 1

# MARL baseline
python3 -m rl.main --env two-jaco-pick-push-place-v0 --gpu 0 --prefix MARL --seed 1 --meta hard --meta_update_target LL --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --diayn False

# MARL with SBD
python3 -m rl.main --env two-jaco-pick-push-place-v0 --gpu 0 --prefix MARL-SBD --seed 1 --meta hard --meta_update_target both --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --max_meta_len 1

# Modular without SBD
python3 -m rl.main --env two-jaco-pick-push-place-v0 --gpu 0 --prefix Modular --seed 1 --meta hard --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --subdiv_skills rl.two-jaco-push-v0.left-n.1/rl.two-jaco-pick-v0.right-n.1,rl.two-jaco-place-v0.right-center-n.1 --max_meta_len 1 --diayn False

# Modular with SBD (Ours)
python3 -m rl.main --env two-jaco-pick-push-place-v0 --gpu 0 --prefix ours --seed 1 --meta hard --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --subdiv_skills rl.two-jaco-push-v0.left-d.1/rl.two-jaco-pick-v0.right-d.1,rl.two-jaco-place-v0.right-center-d.1 --max_meta_len 1
```

  - Ant Push
```bash
# RL baseline
python3 -m rl.main --env ant-push-v0 --gpu 0 --prefix RL --seed 1 --meta hard --meta_update_target LL --subdiv ant_1,box_1,goal_1,ant_2,box_2,goal_2-ant_1,ant_2 --diayn False --max_global_step 4000000 --ckpt_interval 500

# RL with SBD
python3 -m rl.main --env ant-push-v0 --gpu 0 --prefix RL-SBD --seed 1 --meta hard --meta_update_target both --subdiv ant_1,box_1,goal_1,ant_2,box_2,goal_2-ant_1,ant_2 --max_meta_len 5 --max_global_step 4000000 --ckpt_interval 500

# MARL baseline
python3 -m rl.main --env ant-push-v0 --gpu 0 --prefix MARL --seed 1 --meta hard --meta_update_target LL --subdiv ant_1,box_1,goal_1-ant_1/ant_2,box_2,goal_2-ant_2 --diayn False --max_global_step 4000000 --ckpt_interval 500

# MARL with SBD
python3 -m rl.main --env ant-push-v0 --gpu 0 --prefix MARL-SBD --seed 1 --meta hard --meta_update_target both --subdiv ant_1,box_1,goal_1-ant_1/ant_2,box_2,goal_2-ant_2 --max_meta_len 5 --max_global_step 4000000 --ckpt_interval 500

# Modular without SBD
python3 -m rl.main --env ant-push-v0 --gpu 0 --prefix Modular --seed 1 --meta hard --subdiv ant_1,box_1-ant_1/ant_2,box_2-ant_2 --subdiv_skills rl.ant-forward-v0.ant-1-right-n.1,rl.ant-forward-v0.ant-1-left-n.1,rl.ant-forward-v0.ant-1-up-n.1,rl.ant-forward-v0.ant-1-down-n.1/rl.ant-forward-v0.ant-2-right-n.1,rl.ant-forward-v0.ant-2-left-n.1,rl.ant-forward-v0.ant-2-up-n.1,rl.ant-forward-v0.ant-2-down-n.1 --max_meta_len 5 --diayn False --max_global_step 1000000

# Modular with SBD (Ours)
python3 -m rl.main --env ant-push-v0 --gpu 0 --prefix ours --seed 1 --meta hard --subdiv ant_1,box_1-ant_1/ant_2,box_2-ant_2 --subdiv_skills rl.ant-forward-v0.ant-1-right-d.1,rl.ant-forward-v0.ant-1-left-d.1,rl.ant-forward-v0.ant-1-up-d.1,rl.ant-forward-v0.ant-1-down-d.1/rl.ant-forward-v0.ant-2-right-d.1,rl.ant-forward-v0.ant-2-left-d.1,rl.ant-forward-v0.ant-2-up-d.1,rl.ant-forward-v0.ant-2-down-d.1 --max_meta_len 5 --max_global_step 1000000
```


## Directories
The structure of the repository:
- `rl`: Reinforcement learning code
- `env`: Environment code for simulated experiments (Jaco, Ant)
- `util`: Utility code

Log directories:
- `logs/rl.ENV.PREFIX.SEED`:
    - `cmd.sh`: A command used for running a job
    - `git.txt`: Log gitdiff
    - `prarms.json`: Summary of parameters
    - `video`: Generated evaulation videos (every `evalute_interval`)
    - `wandb`: Training summary of W&B, like tensorboard summary
    - `ckpt_*.pt`: Stored checkpoints (every `ckpt_interval`)
    - `replay_*.pt`: Stored replay buffers (every `ckpt_interval`)


## Reference
- PyTorch implementation of PPO and SAC: https://github.com/youngwoon/silo
- TensorFlow implementation of SAC: https://github.com/rail-berkeley/softlearning


## Citation
```
@inproceedings{lee2020learning,
  title={Learning to Coordinate Manipulation Skills via Skill Behavior Diversification},
  author={Youngwoon Lee and Jingyun Yang and Joseph J. Lim},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=ryxB2lBtvH}
}
```

