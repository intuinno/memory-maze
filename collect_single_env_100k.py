from absl import app

from dm_control import viewer
from memory_maze import tasks
import gym
import numpy as np
from tqdm import tqdm
import os

os.environ["MUJOCO_GL"] = "egl"


env = tasks._memory_maze(
    9,  # Maze size
    3,  # n_targets
    1_000_000,  # time_limit
    image_only_obs=False,
    target_color_in_image=False,
    global_observables=True,
    top_camera=True,
    good_visibility=False,
    show_path=False,
    camera_resolution=64,
)


time_step = env.reset()
# Storage for observations and actions
observations = []
actions = []

# Run for 1 million steps
max_steps = 100_000
step = 0

action_spec = env.action_spec()

for step in tqdm(range(max_steps)):
    # Extract observation
    obs = {key: value.copy() for key, value in time_step.observation.items()}
    observations.append(obs)

    # Take a random action
    action = np.random.randint(action_spec.minimum, action_spec.maximum + 1)
    actions.append(action)

    # Step the environment
    time_step = env.step(action)

    # Reset the environment if the episode ends
    if time_step.last():
        time_step = env.reset()

# Convert to NumPy arrays
# Observations need to be handled as a structured array due to multiple keys
obs_keys = observations[0].keys()
obs_array = {key: np.array([obs[key] for obs in observations]) for key in obs_keys}
actions = np.array(actions)

# Save to a .npz file
np.savez("data/single_env_100k.npz", **obs_array, actions=actions)
