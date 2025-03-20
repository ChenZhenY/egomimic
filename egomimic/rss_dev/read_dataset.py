import h5py
import numpy as np
from PIL import Image

dataset_path = "/home/droid_robot/zhenyang/EgoMimic/datasets/lift_all_obs_v141.hdf5"
dataset_path = "/home/droid_robot/Downloads/groceries_robot.hdf5"
dataset_path = "/home/droid_robot/zhenyang/EgoMimic/datasets/rss_sim_datasets/lift.hdf5"

with h5py.File(dataset_path, "r") as f:
    data_attr_keys = list(f.attrs.keys())
    print(f"data_attr_keys: {data_attr_keys}")
    demo_keys = list(f.keys())
    print(demo_keys)
    print(f"data keys: {[demo_keys[0]]}")
    print(f"demo_0 keys: {f['data/demo_0'].keys()}")
    print(f"obs keys: {f['data/demo_0']['obs'].keys()}")
    print("\nDemo Metadata:")
    attr_keys = list(f['data/demo_0'].attrs.keys())
    print(f"Attributes: {attr_keys}") 

    ## customize of the action chunk information
    # print(f"actions shape: {f['data/demo_0']['actions_joints'][:].shape}")
    # print(f"action_chunks shape: {f['data/demo_0']['actions_joints_act'][:].shape}")
    # print(f"actions_joints_act keys: {f['data/demo_0']['obs/front_img_1'][:].shape}")

    ## test and save images
    agentview_image_arr = f['data/demo_0']['obs/agentview_image'][:][10,...]
    print(f"agentview_image.shape: {agentview_image_arr.shape}")
    agentview_image_arr = np.rot90(agentview_image_arr, k=2)
    agentview_image = Image.fromarray(agentview_image_arr)
    agentview_image.save("agentview_image.png")

    robot0_eye_in_hand_image = Image.fromarray(f['data/demo_0']['obs/robot0_eye_in_hand_image'][:][10,...])
    # print(f"robot0_eye_in_hand_image.shape: {robot0_eye_in_hand_image.shape}")
    robot0_eye_in_hand_image.save("robot0_eye_in_hand_image.png")
