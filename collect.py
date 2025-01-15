from absl import app

from dm_control import viewer
from memory_maze import tasks
import gym
import numpy as np
from tqdm import tqdm
import os

os.environ["MUJOCO_GL"] = "egl"


env = tasks.memory_maze_9x9(
    image_only_obs=False,
    target_color_in_image=False,
    global_observables=True,
    top_camera=True,
    good_visibility=False,
    show_path=False,
    camera_resolution=64,
)


def main(unused_argv):
    time_step = env.reset()
    # Storage for observations and actions
    observations = []
    actions = []

    # Run for 1 million steps
    max_steps = 1_000
    step = 0

    for step in tqdm(range(max_steps)):
        # Extract observation
        obs = {key: value.copy() for key, value in time_step.observation.items()}
        observations.append(obs)

        # Take a random action
        action = env.action_spec().generate_value()
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
    np.savez("dm_env_run_data.npz", **obs_array, actions=actions)

    print("Data saved to dm_env_run_data.npz.")

    # viewer.launch(environment_loader=env)


if __name__ == "__main__":
    app.run(main)
