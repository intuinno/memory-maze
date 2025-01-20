from absl import app

from dm_control import viewer
from memory_maze import tasks
import gym
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import pickle

np.random.seed(42)

os.environ["MUJOCO_GL"] = "egl"

# PID Controller Parameters
K_p = 10.0  # Proportional gain
K_i = 0.0  # Integral gain
K_d = 0.0  # Derivative gain

# Target and State Variables
target_angle = np.pi / 2  # Target: 90 degrees (radians)
integral_error = 0.0
previous_error = 0.0

# List to store error values
error_values = []


# Helper Function: Normalize angle to [-π, π]
def normalize_to_pi(angle):
    """
    Normalize an angle to the range [-π, π].

    :param angle: Angle in radians.
    :return: Equivalent angle in the range [-π, π].
    """
    angle = angle % (2 * np.pi)  # Normalize to [0, 2π)
    if angle > np.pi:
        angle -= 2 * np.pi  # Convert [π, 2π) to [-π, π)
    return angle


class Orientation:
    def __init__(self, initial_orientation=0):
        self.angle = initial_orientation

    def turn_left(self):
        self.angle += 1

    def turn_right(self):
        self.angle -= 1

    def get_orientation(self):
        orient = self.angle % 4 * np.pi / 2
        return normalize_to_pi(orient)


# PID Controller Function
def pid_control(current_angle, target_angle, dt):
    """PID controller with fixed positive/negative control signal values."""
    global integral_error, previous_error

    # Compute error
    error = normalize_to_pi(target_angle - current_angle)
    integral_error += error * dt  # Accumulate error over time
    derivative_error = (error - previous_error) / dt  # Rate of change of error
    previous_error = error

    # Store error value
    error_values.append(error)

    # Compute raw PID control signal
    raw_control_signal = K_p * error + K_i * integral_error + K_d * derivative_error

    # Apply fixed control values based on the sign of the control signal
    if raw_control_signal > 0:
        control_signal = 2
    elif raw_control_signal < 0:
        control_signal = 3
    else:
        control_signal = 0.0  # No action if the control signal is exactly zero

    return control_signal


def get_current_angle(agent_dir):
    current_angle = np.arctan2(agent_dir[1], agent_dir[0])
    return current_angle


def get_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two positions.

    :param pos1: First position (x, y).
    :param pos2: Second position (x, y).
    :return: Euclidean distance between pos1 and pos2.
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


env = tasks._memory_maze(
    5,  # Maze size
    1,  # n_targets
    1_000_000,  # time_limit
    image_only_obs=False,
    target_color_in_image=False,
    global_observables=True,
    top_camera=True,
    good_visibility=False,
    show_path=False,
    camera_resolution=64,
    control_freq=4,
    seed=42,
)


time_step = env.reset()
# Storage for observations and actions
observations = []
actions = []
top_action = []

# Run for 1 million steps
max_steps = 100000
step = 0

action_spec = env.action_spec()

orient = Orientation(0)

action_distribution = [1, 1, 2, 3]

for step in tqdm(range(max_steps)):
    # Extract observation
    obs = {key: value.copy() for key, value in time_step.observation.items()}
    obs['orientation'] = orient.angle 

    observations.append(obs)

    current_pos = obs["agent_pos"]

    # Take a random action
    # action = np.random.randint(action_spec.minimum, action_spec.maximum + 1)
    # action = determine_action[step]
    action = np.random.choice(action_distribution)

    sub_action = []

    if action == 0:
        pass
    elif action == 1:
        actions.append(1)
        for k in range(1):
            sub_action.append(k)
            time_step = env.step(1)
        for k in range(3):
            sub_action.append(k)
            time_step = env.step(0)
            # obs = {key: value.copy() for key, value in time_step.observation.items()}
            # observations.append(obs)
    elif action == 2:
        actions.append(2)
        current_angle = get_current_angle(obs["agent_dir"])
        orient.turn_left()
        target_angle = orient.get_orientation()
        while True:
            previous_angle = current_angle
            dt = env.control_timestep()
            s_action = pid_control(current_angle, target_angle, dt)
            sub_action.append(s_action)
            time_step = env.step(s_action)
            # obs = {key: value.copy() for key, value in time_step.observation.items()}
            # observations.append(obs)
            if time_step.last():
                break
            current_angle = get_current_angle(time_step.observation["agent_dir"])
            if abs(normalize_to_pi(current_angle - target_angle)) < 0.05:
                break
            elif abs(normalize_to_pi(previous_angle - current_angle)) < 0.01:
                break
            elif len(sub_action) > 100:
                print(
                    "Timeout at",
                    step,
                    action,
                    sub_action,
                    previous_angle,
                    current_angle,
                )
                break

    elif action == 3:
        actions.append(3)
        current_angle = get_current_angle(obs["agent_dir"])
        orient.turn_right()
        target_angle = orient.get_orientation()
        while True:
            previous_angle = current_angle
            dt = env.control_timestep()
            s_action = pid_control(current_angle, target_angle, dt)
            sub_action.append(s_action)
            time_step = env.step(s_action)
            # obs = {key: value.copy() for key, value in time_step.observation.items()}
            # observations.append(obs)
            if time_step.last():
                break
            current_angle = get_current_angle(time_step.observation["agent_dir"])
            if abs(normalize_to_pi(current_angle - target_angle)) < 0.05:
                break
            elif abs(normalize_to_pi(previous_angle - current_angle)) < 0.01:
                break
            elif len(sub_action) > 100:
                print(
                    "Timeout at",
                    step,
                    action,
                    sub_action,
                    previous_angle,
                    current_angle,
                )
                break

    # Step the environment
    # time_step = env.step(action)
    top_action.append({action: sub_action})

    # Reset the environment if the episode ends
    if time_step.last():
        time_step = env.reset()

    if (
        action == 1
        and get_distance(current_pos, time_step.observation["agent_pos"]) < 0.1
    ):
        action_distribution = [2, 3]
    else:
        action_distribution = [1, 1, 2, 3]


# Convert to NumPy arrays
# Observations need to be handled as a structured array due to multiple keys
obs_keys = observations[0].keys()
obs_array = {key: np.array([obs[key] for obs in observations]) for key in obs_keys}
actions = np.array(actions)
error_values = np.array(error_values)

# Save to a .npz file
np.savez(
    "data/small_env_5_5_3actions_100k_low_orient.npz",
    **obs_array,
    actions=actions,
    error_values=error_values,
)

# Save top_action as a pickle file
with open("data/top_action_100k_orient.pkl", "wb") as f:
    pickle.dump(top_action, f)

# Plot the error signal over timesteps
# plt.figure()
# plt.plot(error_values)
# plt.xlabel("Timestep")
# plt.ylabel("Error")
# plt.title("PID Error Signal Over Time")
# plt.show()
# 
# # Plot the trajectory using obs['agent_pos']
# agent_positions = np.array([obs["agent_pos"] for obs in observations])
# 
# plt.figure()
# plt.plot(agent_positions[:, 0], agent_positions[:, 1], marker="o")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Agent Trajectory")
# plt.show()
