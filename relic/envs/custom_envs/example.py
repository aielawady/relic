import gym
from gym import spaces
import numpy as np
from habitat.core.dataset import BaseEpisode

from relic.envs.custom_envs.darkroom import DarkroomEnv

try:
    import gymnasium

    IS_GYMNASIUM_AVAILABLE = True
except ImportError:
    IS_GYMNASIUM_AVAILABLE = False


def b2b(src: "gymnasium.spaces.Box"):
    """Converts gymnasium.spaces.Box to gym.spaces.Box"""
    try:
        return spaces.Box(low=src.low, high=src.high, dtype=src.dtype, shape=src.shape)
    except Exception:
        return spaces.Box(low=src.low, high=src.high, shape=src.shape)


def d2d(src: "gymnasium.spaces.Discrete"):
    """Converts gymnasium.spaces.Discrete to gym.spaces.Discrete"""
    return spaces.Discrete(src.n, start=src.start)


def convert2gym_space(src):
    """Converts from gymnasium spaces to gym spaces."""
    if isinstance(src, (spaces.Box, spaces.Discrete)):
        return src
    elif IS_GYMNASIUM_AVAILABLE:
        if isinstance(src, (gymnasium.spaces.Box)):
            return b2b(src)
        elif isinstance(src, gymnasium.spaces.Discrete):
            return d2d(src)
        else:
            raise TypeError(f"The conversion is not implement for type {type(src)}.")
    else:
        raise TypeError(
            f"The conversion is not implement for type {type(src)}. If this is a gymnasium class, make sure the package is installed."
        )


class NewEnv(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        """Create env wrapper compatible with ReLIC."""

        # Each element of self.goals is a different task in the darkroom env.
        # This can be replaced by seed in gym/gymnasium envs which is passed to reset calls.
        self.goals = kwargs.pop("goals")
        super().__init__(*args, **kwargs)

        # ReLIC supports gym spaces. This is how to convert from gymnasium spaces to gym spaces.
        self.observation_space = spaces.Dict(
            {
                "state": convert2gym_space(self.env.observation_space),
                "reward_input": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )

        # ReLIC supports gym spaces. This is how to convert from gymnasium spaces to gym spaces.
        self.action_space = convert2gym_space(self.env.action_space)

        # self.current_goal_idx indicates the trial's goal so that when the env is reset in the trial
        # the goal doesn't change.
        self.current_goal_idx = 0
        self.after_update()

        # Required information for the evaluator.
        self.episodes = [str(x) for x in self.goals]
        self.number_of_episodes = len(self.episodes)
        self._has_number_episode = True

    def step(self, action):
        # relic passes int for categorical actions but darkroom accepts 1-hot encoding.
        new_action = np.zeros(self.action_space.n)
        new_action[action] = 1

        obs, reward, done, info = super().step(new_action)
        obs = {
            "state": obs,
            "reward_input": [reward],
        }
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        """Reset the env.

        This method shoud reset the env but not change the task. This can be
        controlled by using the same goal in the darkroom env or the same seed
        in gym/gymnasium envs. The method responsible for changing the task is
        `after_update`.
        """
        obs = super().reset(goal=self.goals[self.current_goal_idx])
        obs = {
            "state": obs,
            "reward_input": [0],
        }
        return obs

    def after_update(self):
        """Change the task. This method is called after the trial ends to change the task."""
        self.current_goal_idx += 1
        self.current_goal_idx %= len(self.goals)

    def current_episode(self, *args):
        """Return episode identifier."""
        goal = self.goals[self.current_goal_idx]
        return BaseEpisode(
            episode_id=str(goal),
            scene_id=str(goal),
        )

    @property
    def original_action_space(self) -> spaces.space:
        """Return the action space."""
        return self.action_space


def make_env(config, dataset=None):
    env = DarkroomEnv(10, (0, 0), 100)

    # Create train/validation splits
    is_eval = config.habitat_baselines.evaluate
    goals = np.array([[(j, i) for i in range(10)] for j in range(10)]).reshape(-1, 2)
    np.random.RandomState(seed=0).shuffle(goals)
    train_test_split = int(0.8 * len(goals))
    if is_eval:
        goals = goals[train_test_split:]
    else:
        goals = goals[:train_test_split]

    # Shuffle the tasks. config.habitat.seed is different for each env worker.
    np.random.RandomState(config.habitat.seed).shuffle(goals)

    return NewEnv(env, goals=goals)
