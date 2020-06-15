import numpy as np
from lunzi.dataset import Dataset
from slbo.envs import BaseBatchedEnv
from slbo.policies import BasePolicy
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.v_function import BaseVFunction
import tensorflow as tf
import sys
from slbo.utils.flags import FLAGS

class Runner(object):
    _states: np.ndarray  # [np.float]
    _n_steps: np.ndarray
    _returns: np.ndarray

    def __init__(self, env: BaseBatchedEnv, max_steps: int, gamma=0.99, lambda_=0.95, rescale_action=False):
        self.env = env
        self.n_envs = env.n_envs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_steps = max_steps
        self.rescale_action = rescale_action
        self._dtype = gen_dtype(env, 'state action next_state reward done timeout')

        self.reset()

    def reset(self):
        self.set_state(self.env.reset(), set_env_state=False)

    def set_state(self, states: np.ndarray, set_env_state=True):
        self._states = states.copy()
        if set_env_state:
            self.env.set_state(states)
        self._n_steps = np.zeros(self.n_envs, 'i4')
        self._returns = np.zeros(self.n_envs, 'f8')

    def get_state(self):
        return self._states.copy()

    def run(self, policy: BasePolicy, n_samples: int):
        ep_infos = []
        self.rewards_params_list = []
        self.reward_ctrl_list = []
        self.reward_state_list = []
        n_steps = n_samples // self.n_envs
        assert n_steps * self.n_envs == n_samples
        dataset = Dataset(self._dtype, n_samples)
        self.begin_mark = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        start = np.array([0 for _ in range(self.n_envs)])

        for T in range(n_steps):
            unscaled_actions = policy.get_actions(self._states)
            if self.rescale_action:
                lo, hi = self.env.action_space.low, self.env.action_space.high
                actions = (lo + (unscaled_actions + 1.) * 0.5 * (hi - lo))
            else:
                actions = unscaled_actions

            next_states, rewards, dones, infos = self.env.step(actions)
            self.reward_ctrl_list.append([i['reward_ctrl'] for i in infos])
            self.reward_state_list.append([i['reward_state'] for i in infos])
            dones = dones.astype(bool)
            self._returns += rewards
            self._n_steps += 1
            timeouts = self._n_steps == self.max_steps

            steps = [self._states.copy(), unscaled_actions, next_states.copy(), rewards, dones, timeouts]
            dataset.extend(np.rec.fromarrays(steps, dtype=self._dtype))

            indices = np.where(dones | timeouts)[0]
            if len(indices) > 0:
                next_states = next_states.copy()
                next_states[indices] = self.env.partial_reset(indices)
                for index in indices:
                    infos[index]['episode'] = {'return': self._returns[index]}
                    self.begin_mark[start[index]][index] = 1
                self._n_steps[indices] = 0
                self._returns[indices] = 0.
                start[indices] = T+1

            self._states = next_states.copy()
            ep_infos.extend([info['episode'] for info in infos if 'episode' in info])

        if len(ep_infos) == 0:
            print ("oops!")
            assert(False)
        return dataset, ep_infos

    def compute_advantage(self, vfn: BaseVFunction, samples: Dataset, task=None):
        n_steps = len(samples) // self.n_envs
        samples = samples.reshape((n_steps, self.n_envs))
        use_next_vf = ~samples.done
        use_next_adv = ~(samples.done | samples.timeout)

        next_values = vfn.get_values(samples[-1].next_state)
        values = vfn.get_values(samples.reshape(-1).state).reshape(n_steps, self.n_envs)
        advantages = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        advantages_shadow = np.zeros((n_steps, self.n_envs), dtype=np.float32)

        next_values_all = np.zeros_like(values, dtype=np.float32)
        next_values_all[:-1] = values[1:] * (1.0 - samples.done[1:])
        next_values_all[-1] = next_values
        td = self.gamma * next_values_all * use_next_vf - values

        coef_mat = np.zeros([n_steps, n_steps, self.n_envs], np.float32)
        coef_mat_returns = np.zeros([n_steps, n_steps, self.n_envs], np.float32)
        #print ('use_next_adv:', use_next_adv.shape)
        tmp = []
        for i in range(n_steps):
            coef = np.ones([self.n_envs], dtype=np.float32)
            coef_r = np.ones([self.n_envs], dtype=np.float32)
            coef_mat[i][i] = coef
            coef_mat_returns[i][i] = coef_r
            if i == n_steps-1: tmp.append(coef)
            for j in range(i+1, n_steps):
                coef *= self.gamma * self.lambda_ * use_next_adv[j-1] #~samples.done[j] #* use_next_adv[j] #~samples.done[j]
                if i == n_steps-1: tmp.append(coef)
                coef_mat[i][j] = coef
                #TODO
                coef_r *= self.gamma * use_next_vf[j-1] #~samples.done[j]
                coef_mat_returns[i][j] = coef_r
        coef_mat = np.transpose(coef_mat, (2, 0, 1))
        coef_mat_returns = np.transpose(coef_mat_returns, (2, 0, 1))

        reward_ctrl_list = np.array(self.reward_ctrl_list, dtype=np.float32)
        reward_state_list = np.array(self.reward_state_list, dtype=np.float32)

        last_gae_lambda = 0
        next_values = vfn.get_values(samples[-1].next_state)
        for t in reversed(range(n_steps)):
            delta =  samples[t].reward + self.gamma * next_values * use_next_vf[t] - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_ * last_gae_lambda * use_next_adv[t]
            next_values = values[t]

        advantages_params = None
        return advantages.reshape(-1), advantages_params, values.reshape(-1), td, coef_mat, coef_mat_returns, reward_ctrl_list, reward_state_list, self.begin_mark
