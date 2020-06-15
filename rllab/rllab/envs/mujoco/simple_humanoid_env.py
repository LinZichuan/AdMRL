from rllab.envs.base import Step
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import autoargs

from gym import utils
from gym.envs.mujoco import mujoco_env

class SimpleHumanoidEnv(mujoco_env.MujocoEnv, Serializable):

    FILE = 'simple_humanoid.xml'

    @autoargs.arg('vel_deviation_cost_coeff', type=float,
                  help='cost coefficient for velocity deviation')
    @autoargs.arg('alive_bonus', type=float,
                  help='bonus reward for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for control inputs')
    @autoargs.arg('impact_cost_coeff', type=float,
                  help='cost coefficient for impact')
    def __init__(
            self,
            n_tasks=100,
            randomize_tasks=False,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-3,
            impact_cost_coeff=1e-5,
            *args, **kwargs):
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 1)
        utils.EzPickle.__init__(self)
        #super(SimpleHumanoidEnv, self).__init__(*args, **kwargs)
        #Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return self._get_obs()

    def _get_obs(self):
        data = self.sim.data
        #print (data.qpos.shape)
        #print (data.qvel.shape)
        #print (self.get_body_com("torso").shape)
        #print (self.get_body_comvel("torso").shape)
        return np.concatenate([
            data.qpos.flat,  # 17
            data.qvel.flat,  # 16
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    #def _get_com(self):
    #    data = self.sim.data
    #    mass = self.model.body_mass
    #    xpos = data.xipos
    #    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def reset_model(self):
        self.comvel = np.array([0., 0., 0.])
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    @property
    def action_bounds(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        return bounds.T

    def get_body_comvel(self, body_name):
        return self.comvel

    def forward_dynamics(self, action):
        xposbefore = np.copy(self.get_body_com('torso'))
        self.do_simulation(action, 1)
        xposafter = self.get_body_com('torso')
        #print (xposbefore, xposafter, self.dt)
        self.comvel = (xposafter - xposbefore) / self.dt
        #print ("comvel:", self.comvel)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        alive_bonus = self.alive_bonus
        data = self.sim.data

        comvel = self.get_body_comvel("torso")

        lin_vel_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = .5 * self.impact_cost_coeff * np.sum(
            np.square(np.clip(data.cfrc_ext, -1, 1)))
        vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))
        reward = lin_vel_reward + alive_bonus - ctrl_cost - \
            impact_cost - vel_deviation_cost
        done = data.qpos[2] < 0.8 or data.qpos[2] > 2.0

        return Step(next_obs, reward, done)

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
