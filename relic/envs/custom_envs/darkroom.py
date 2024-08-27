# The code from https://github.com/jon--lee/decision-pretrained-transformer/blob/15172df758a04bb112fa949dd62dcb9765780ec0/envs/darkroom_env.py#L12-L82
# With change in reset method to accept `goal` argument.

import gym
import numpy as np


class DarkroomEnv(gym.Env):
    def __init__(self, dim, goal, horizon):
        self.dim = dim
        self.goal = np.array(goal)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 5
        self.observation_space = gym.spaces.Box(
            low=0, high=dim - 1, shape=(self.state_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.action_dim)

    def sample_state(self):
        return np.random.randint(0, self.dim, 2)

    def sample_action(self):
        i = np.random.randint(0, 5)
        a = np.zeros(self.action_space.n)
        a[i] = 1
        return a

    def reset(self, goal=None):
        if goal is not None:
            self.goal = goal

        self.current_step = 0
        self.state = np.array([0, 0])
        return self.state

    def transit(self, state, action):
        action = np.argmax(action)
        assert action in np.arange(self.action_space.n)
        state = np.array(state)
        if action == 0:
            state[0] += 1
        elif action == 1:
            state[0] -= 1
        elif action == 2:
            state[1] += 1
        elif action == 3:
            state[1] -= 1
        state = np.clip(state, 0, self.dim - 1)

        if np.all(state == self.goal):
            reward = 1
        else:
            reward = 0
        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = self.current_step >= self.horizon
        return self.state.copy(), r, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        if state[0] < self.goal[0]:
            action = 0
        elif state[0] > self.goal[0]:
            action = 1
        elif state[1] < self.goal[1]:
            action = 2
        elif state[1] > self.goal[1]:
            action = 3
        else:
            action = 4
        zeros = np.zeros(self.action_space.n)
        zeros[action] = 1
        return zeros
