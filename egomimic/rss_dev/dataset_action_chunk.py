"""
change action to action_chunk for robomimic dataset
"""

import h5py
import numpy as np


# dataset_path = "/home/droid_robot/zhenyang/EgoMimic/datasets/lift_all_obs_v141.hdf5"
# dataset_path = "/home/droid_robot/zhenyang/EgoMimic/datasets/lift_all_obs_v141.hdf5"
dataset_path = "/home/droid_robot/zhenyang/EgoMimic/datasets/rss_sim_datasets/square_final_dataset.hdf5"
single_action_key = "absolute_actions_with_precision"
action_chunk_key = "absolute_actions_with_precision_act"
action_chunk_size = 16


def create_action_chunk(single_action_key, action_chunk_key, action_chunk_size):
    """
    action_chunk: [current_action, current_action + action_chunk_size - 1]
    for last few action, we pad with the same actions.
    """
    with h5py.File(dataset_path, "r+") as f:
        demo_keys = list(f['data'].keys())

        for demo_key in demo_keys:
            single_actions = f['data'][demo_key][single_action_key][:] # time_len X action_dim
            time_len, action_dim = single_actions.shape

            # pad actions to make single_actions as action_chunk_size*n X action_dim
            print(f"demo_key: {demo_key} *****************")
            print(f"single_actions shape: {single_actions.shape}")
            pad_actions = np.tile(single_actions[-1], (action_chunk_size, 1))
            single_actions = np.concatenate([single_actions, pad_actions], axis=0)
            print(f"padded single_actions shape: {single_actions.shape}")

            # action chunk
            action_chunks = np.zeros((time_len, action_chunk_size, action_dim))  
            for i in range(time_len):
                action_chunks[i, :, :] = single_actions[i:i+action_chunk_size, :]

            print(f"action_chunks shape: {action_chunks.shape}")
            f['data'][demo_key][action_chunk_key] = action_chunks


if __name__ == "__main__":
    create_action_chunk(single_action_key, action_chunk_key, action_chunk_size)



