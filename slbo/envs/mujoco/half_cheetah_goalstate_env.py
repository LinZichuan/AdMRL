import math
import numpy as np
from rllab.envs.mujoco import half_cheetah_goalstate_env
from slbo.envs import BaseModelBasedEnv
from slbo.utils.flags import FLAGS

HalfCheetahGoalStateConfig = half_cheetah_goalstate_env.HalfCheetahGoalStateConfig

class HalfCheetahGoalStateEnv(half_cheetah_goalstate_env.HalfCheetahGoalStateEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return self._get_obs()

    def mb_step(self, states, actions, next_states):
        #print (self.get_current_obs(), next_states)
        actions = np.clip(actions, *self.action_bounds)
        reward_ctrl = -0.05 * np.sum(np.square(actions), axis=-1)
        reward_ctrl = reward_ctrl.astype(np.float32)
        norm_next_states = (next_states - self._task_config.s_mean) / self._task_config.s_std
        reward_run = np.sum(-np.abs(norm_next_states - self._task_config.goal_velocity) * self._task_config.coef, axis=1)
        reward_state = norm_next_states

        reward = reward_ctrl + reward_run
        #return reward_ctrl + reward_fwd, np.zeros_like(reward_fwd, dtype=np.bool), reward_ctrl, reward_state.reshape((-1, 1))
        return reward, np.zeros_like(reward_run, dtype=np.bool), reward_ctrl, reward_state
