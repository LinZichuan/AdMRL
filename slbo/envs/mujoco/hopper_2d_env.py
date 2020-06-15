import numpy as np
from rllab.envs.mujoco import hopper_2d_env
from rllab.envs.base import Step
from slbo.envs import BaseModelBasedEnv

Hopper2DConfig = hopper_2d_env.Hopper2DConfig

class Hopper2DEnv(hopper_2d_env.Hopper2DEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return self._get_obs()

    def mb_step(self, states, actions, next_states):
        # reward_ctrl
        reward_ctrl = - 0.001 * np.sum(np.square(actions), axis=-1) + self._task_config.alive_bonus
        # reward_run
        height, ang, vel = next_states[...,0], next_states[...,1], next_states[...,5]
        #height = next_states[:, 0]
        #vel = next_states[:, -3]
        reward_run = -np.abs(vel - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        reward_run += -np.abs(height - self._task_config.goal_velocity[1]) * self._task_config.coef[1]
        state1 = np.array(vel).reshape((-1,1))
        state2 = np.array(height).reshape((-1,1))
        reward_state = np.concatenate([state1, state2], axis=-1)

        # reward, done, reward_state
        reward = reward_ctrl + reward_run
        done = ~((next_states[:, 2:] < 100).all(axis=-1) &
                 (height > 0.7) &
                 (np.abs(ang) < 0.2))
        #print (reward_state.shape)
        return reward, done, reward_ctrl, reward_state
