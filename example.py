from absl import app

from dm_control import viewer
from memory_maze import tasks
import gym

env = tasks.memory_maze_15x15(
    # global_observables=True,
    control_freq=20,
    walker_str="ant",
    discrete_actions=False,
    target_color_in_image=False,
    remap_obs=False,
    bonus_time_limit=100000000000,
)


def main(unused_argv):
    gym_env = gym.make("memory_maze:MemoryMaze-15x15-Ant-v0")
    print("observation space\n", gym_env.observation_space)
    print("action_space\n", gym_env.action_space)
    viewer.launch(environment_loader=env)


if __name__ == "__main__":
    app.run(main)
