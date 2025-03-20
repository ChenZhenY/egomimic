import h5py
import numpy as np


dataset_path = "/home/droid_robot/zhenyang/EgoMimic/datasets/lift_all_obs_v141.hdf5"
dataset_path = "/home/droid_robot/Downloads/groceries_robot.hdf5"
dataset_path = "/home/droid_robot/zhenyang/EgoMimic/datasets/rss_sim_datasets/can.hdf5"

with h5py.File(dataset_path, "r") as f:
    demo_keys = list(f.keys())
    print(demo_keys)
    print(f"data keys: {[demo_keys[0]]}")
    print(f"demo_0 keys: {f['data/demo_0'].keys()}")
    print(f"obs keys: {f['data/demo_0']['obs'].keys()}")

    ## customize of the action chunk information
    print(f"actions shape: {f['data/demo_0']['actions_joints'][:].shape}")
    print(f"action_chunks shape: {f['data/demo_0']['actions_joints_act'][:].shape}")