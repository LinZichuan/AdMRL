import numpy as np
from rllab.envs.mujoco import walker_2d_env
from rllab.envs.base import Step
from slbo.envs import BaseModelBasedEnv

Walker2DConfig = walker_2d_env.Walker2DConfig

class Walker2DEnv(walker_2d_env.Walker2DEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return self._get_obs()

    def mb_step(self, states, actions, next_states):
        # reward_ctrl
        reward_ctrl = -0.001 * np.sum(np.square(actions), axis=-1) + self._task_config.alive_bonus
        # reward_run
        height, ang, vel = next_states[...,0], next_states[...,1], next_states[...,8]
        reward_run = -np.abs(vel - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        reward_run += -np.abs(height - self._task_config.goal_velocity[1]) * self._task_config.coef[1]
        state1 = np.array(vel).reshape((-1,1))
        state2 = np.array(height).reshape((-1,1))
        reward_state = np.concatenate([state1, state2], axis=-1)
        #print (reward_state.shape)

        # reward, done, reward_state
        rewards = reward_ctrl + reward_run

        dones = ~((height > 0.8) &
                  (height < 2.0) &
                  (ang > -1.0) &
                  (ang < 1.0))
        return rewards, dones, reward_ctrl, reward_state
