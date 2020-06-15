import math
import numpy as np
from rllab.envs.mujoco import half_cheetah_2d_env
from slbo.envs import BaseModelBasedEnv
from slbo.utils.flags import FLAGS

HalfCheetah2DConfig = half_cheetah_2d_env.HalfCheetah2DConfig

class HalfCheetah2DEnv(half_cheetah_2d_env.HalfCheetah2DEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return self._get_obs()

    def mb_step(self, states, actions, next_states):
        # reward_ctrl
        reward_ctrl = -0.1 * np.sum(np.square(actions), axis=-1)

        # reward_run
        height, vel = next_states[...,0], next_states[...,self.qpos_len]
        reward_run = -np.abs(vel - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        reward_run += -np.abs(height - self._task_config.goal_velocity[1]) * self._task_config.coef[1]

        # reward, done, reward_state
        reward = reward_ctrl + reward_run
        dones = np.zeros_like(reward_run, dtype=np.bool)
        reward_state = np.concatenate([vel, height], axis=-1)
        return reward, dones, reward_ctrl, reward_state.reshape((-1, 1))
