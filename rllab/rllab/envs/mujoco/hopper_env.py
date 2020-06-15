import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
#from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides

from gym import utils
from gym.envs.mujoco import mujoco_env

# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HopperEnv(mujoco_env.MujocoEnv, Serializable):

    FILE = 'hopper.xml'

    def __init__(self, alive_coeff=1, ctrl_cost_coeff=0.01, *args, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 1)
        utils.EzPickle.__init__(self)
        #self.alive_coeff = alive_coeff
        #self.ctrl_cost_coeff = ctrl_cost_coeff
        #super(HopperEnv, self).__init__(*args, **kwargs)
        #Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return self._get_obs()

    @overrides
    def _get_obs(self):
        return None

    def reset_model(self):
        self.comvel = np.array([0., 0., 0.])
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    @property
    def action_bounds(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        return bounds.T

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.sim.data.xmat[idx].reshape((3, 3))

    def get_body_comvel(self, body_name):
        return self.comvel

    def forward_dynamics(self, action):
        xposbefore = np.copy(self.get_body_com('torso'))
        self.do_simulation(action, 1)
        xposafter = self.get_body_com('torso')
        #print (xposbefore, xposafter, self.dt)
        self.comvel = (xposafter - xposbefore) / self.dt
        #print ("comvel:", self.comvel)

    @overrides
    def step(self, action):
        return None
        '''
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = self.get_body_comvel("torso")[0]
        reward = vel + self.alive_coeff - \
            0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        state = self._state
        notdone = np.isfinite(state).all() and \
            (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
            (abs(state[2]) < .2)
        done = not notdone
        return Step(next_obs, reward, done)
        '''

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
