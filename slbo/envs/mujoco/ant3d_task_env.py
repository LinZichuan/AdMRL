import math
import numpy as np
from rllab.envs.mujoco import ant3d_task_env
from rllab.envs.base import Step
from slbo.envs import BaseModelBasedEnv
from slbo.utils.flags import FLAGS

Ant3DTaskConfig = ant3d_task_env.Ant3DTaskConfig

class Ant3DTaskEnv(ant3d_task_env.Ant3DTaskEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return self._get_obs()


    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        comvel = next_states[..., -3:]
        #zpos = next_states[..., -4]
        zpos = next_states[..., 2]

        forward_reward = -np.abs(comvel[..., 0] - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        forward_reward += -np.abs(comvel[..., 1] - self._task_config.goal_velocity[1]) * self._task_config.coef[1]
        forward_reward += -np.abs(zpos - self._task_config.goal_velocity[2]) * self._task_config.coef[2]

        state1 = np.array(comvel[..., 0]).reshape((-1, 1))
        state2 = np.array(comvel[..., 1]).reshape((-1, 1))
        state3 = np.array(zpos).reshape((-1, 1))
        reward_state = np.concatenate([state1, state2, state3], axis=-1) #nenv, 2
        entry = 3.0
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(actions / scaling), axis=-1)
        contact_cost = 0.
        survive_reward = 0.05
        reward = forward_reward - (ctrl_cost + contact_cost - survive_reward - entry)
        ctrl_cost = ctrl_cost + contact_cost - survive_reward - entry
        notdone = np.all([next_states[..., 2] >= 0.2, next_states[..., 2] <= 1.0], axis=0)
        return reward, 1. - notdone, -ctrl_cost, reward_state

