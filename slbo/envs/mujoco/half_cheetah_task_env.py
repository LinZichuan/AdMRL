import math
import numpy as np
from rllab.envs.mujoco import half_cheetah_task_env
from slbo.envs import BaseModelBasedEnv
from slbo.utils.flags import FLAGS

HalfCheetahTaskConfig = half_cheetah_task_env.HalfCheetahTaskConfig

class HalfCheetahTaskEnv(half_cheetah_task_env.HalfCheetahTaskEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 9
            self.model.data.qvel.flat,  # 9
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def mb_step(self, states, actions, next_states):
        actions = np.clip(actions, *self.action_bounds)
        reward_ctrl = -0.05 * np.sum(np.square(actions), axis=-1)
        reward_ctrl = reward_ctrl.astype(np.float32)
        #print (reward_ctrl.dtype)
        if self._task_config.goal_velocity == -math.inf:
            reward_fwd = -1 * next_states[..., 21]
        elif self._task_config.goal_velocity == math.inf:
            reward_fwd = next_states[..., 21]
        else:
            if FLAGS.task.reward == 'abs':
                reward_fwd = -1. * np.abs(next_states[..., 21] - self._task_config.goal_velocity)
            elif FLAGS.task.reward == 'quadratic':
                reward_fwd = -1. * np.square(next_states[..., 21] - self._task_config.goal_velocity)
            else:
                raise Exception(f'{FLAGS.task.reward} reward function is not available!')
            #print(reward_fwd.dtype)
        reward_state = next_states[..., 21]
        #print (reward_state.shape)
        #NOTE: in original code, reward_ctrl is float64, but reward_fwd is float32 !!!
        return reward_ctrl + reward_fwd, np.zeros_like(reward_fwd, dtype=np.bool), reward_ctrl, reward_state.reshape((-1, 1))
