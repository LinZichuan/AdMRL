import numpy as np
import lunzi.dataset as dataset
import gym

def gen_dtype(env: gym.Env, fields: str):
    dtypes = {
        'state': ('state', 'f4', env.observation_space.shape),
        'action': ('action', 'f4', env.action_space.shape),
        'next_state': ('next_state', 'f4', env.observation_space.shape),
        'reward': ('reward', 'f4'),
        'done': ('done', 'bool'),
        'timeout': ('timeout', 'bool'),
        'return_': ('return_', 'f4'),
        'advantage': ('advantage', 'f4'),
    }
    return [dtypes[field] for field in fields.split(' ')]


class Dataset(dataset.Dataset):
    def sample_multi_step(self, size: int, n_env: int, n_step=1):
        starts = np.random.randint(0, self._len, size=size)
        batch = []
        for step in range(n_step):
            batch.append(self[(starts + step * n_env) % self._len])
        return np.concatenate(batch).reshape(n_step, size).view(np.recarray)

