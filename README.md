# EgoMimic: Scaling Imitation Learning through Egocentric Video
Codebase for EgoMimic.  This repo contains the training code for EgoMimic.  To rollout policies in the real world, you'll additionally need our hardware repo [Eve](https://github.com/SimarKareer/Eve).

[Project Website](https://egomimic.github.io/)

## Installation

```
git clone --recursive git@github.com:SimarKareer/EgoMimic.git
cd EgoMimic
conda env create -f environment.yaml
pip install projectaria-tools'[all]'
pip install -e external/robomimic
pip install -e .
python external/robomimic/robomimic/scripts/setup_macros.py
```

Set `git config --global submodule.recurse true` if you want `git pull` to automatically update the submodule as well.

Then go to  `external/robomimic/robomimic/macros_private.py` and manually add your wandb username. Make sure you have ran `wandb login` too.


**Download Sample Data @Dhruv**
```

```

-------


## EgoMimic Quick Start (Train on Sample Data)

EgoMimic Training (Toy in Bowl Task)
```
python scripts/pl_train.py --config configs/egomimic_oboo.json --dataset /path/to/robot_oboo.hdf5 --dataset_2 /path/to/hand_oboo.hdf5 --debug
```

ACT Baseline Training
```
python scripts/pl_train.py --config configs/act.json --dataset /path/to/robot_oboo.hdf5 --debug
```

For a detailed list of commands to run each experiment see [experiment_launch.md](./experiment_launch.md)

Use `--debug` to check that the pipeline works

Launching runs via submitit / slurm
```
python scripts/pl_submit.py --config <config> --name <name> --description <description> --gpus-per-node <gpus-per-node>`
```

Training creates a folder for each experiment
```
./trained_models_highlevel/description/name
├── videos (generated offline validation videos)
├── logs (wandb logs)
├── slurm (slurm logs if launched via slurm)
├── config.json (copy of config used to launch this run)
├── models (model ckpts)
├── ds1_norm_stats.pkl (robot dataset normalization stats)
└── ds2_norm_stats.pkl (hand data norm stats if training egomimic)
```

Offline Eval:
`python scripts/pl_train.py --dataset <dataset> --ckpt_path <ckpt> --eval`

## Using your own data
### SAM Installation
Processing hand and robot data relies on SAM to generate masks for the hand and robot.  Full SAM [instructions](https://github.com/facebookresearch/segment-anything-2).  

TLDR:
```
cd outsideOfEgomimic
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints && \
./download_ckpts.sh && \
mv sam2_hiera_tiny.pt /path/to/egomimic/resources/sam2_hiera_tiny.pt
```

### Processing Robot Data for Training
To use your own robot, first follow setup instructions in our hardware repo [Eve](https://github.com/SimarKareer/eve).

**Calibrate Cameras**

To train EgoMimic on your own data you must provide the hand-eye-calibration extrinsics matrix inside [``egomimic/utils/egomimicUtils``](./egomimic/utils/egomimicUtils.py)
- Print a large april tag and tape it to the wrist camera mount
- Collect calibration data for each arm one at a time.  Move the arm in many directions for best results.  This will generate an hdf5 under the `data` directory
```
cd eve
python scripts/record_episides.py --task_name CALIBRATE --arm <left or right>

cd egomimic/scripts/calibrate_camera
python calibrate_egoplay.py --h5py-path <path to hdf5 from previous section.hdf5>
```
Paste this matrix into [``egomimic/utils/egomimicUtils``](./egomimic/utils/egomimicUtils.py) for the appropriate arm

**Recording Demos**

```
cd eve
setup_eve
ros_eve
sh scripts/auto_record.sh <task defined in constants.py> <num_demos> <arm: left, right, bimanual> <starting_index>
```
This creates a folder with many demos in it
```
├── TASK_NAME
│   ├── episode_1.hdf5
...
│   ├── episode_n.hdf5
```

**Process Robot Demos**

To process the demos we've recorded we run.  Here's an example command
```
cd egomimic/scripts/aloha_process
python aloha_to_robomimic.py --dataset /path/to/TASK_NAME --arm <left, right, bimanual> --out /desired/output/path.hdf5 --extrinsics <keyName in egoMimicUtils.py>
```

### Process Aria Data for Training
Collect Aria demonstrations via the Aria App, then transfer them to your computer, make the following structure
```
TASK_NAME_ARIA/
├── rawAria
│   ├── demo1.vrs
│   ├── demo1.vrs.json
...
│   ├── demon.vrs
│   ├── demon.vrs.json
└── converted (empty folder)
```

This will process your aria data into hdf5 format, and optionally with the `--mask` argument, it will also use SAM to mask out the hand data.
```
cd egomimic
python scripts/aria_process/aria_to_robomimic.py --dataset /path/to/TASK_NAME_ARIA --out /path/to/converted/TASK_NAME.hdf5 --hand <left, right, or bimanual> --mask
```

### Rollout policies in the real world
Follow these instructions on the desktop connected to the real hardware.
1. Follow instructions in [Eve](https://github.com/SimarKareer/Eve)
2. Install the hardware package into the `emimic` conda env via
```
conda activate emimic
cd ~/interbotix_ws/src/eve
pip install -e .
```
3. Rollout policy
```
cd EgoMimic/egomimic
python scripts/evaluation/eval_real --eval-path <path to>EgoPlay/trained_models_highlevel/<your model folder>/models/<your ckpt>.ckpt
```
