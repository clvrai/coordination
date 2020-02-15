from collections import OrderedDict

import numpy as np

from env.ant.ant import AntEnv
from env.transform_utils import up_vector_from_quat, forward_vector_from_quat, \
    l2_dist, cos_dist, right_vector_from_quat, sample_quat


class AntForwardEnv(AntEnv):
    def __init__(self, **kwargs):
        self.name = 'ant_forward'
        super().__init__('ant_forward.xml', **kwargs)

        # Env config
        self._env_config.update({
            'vel_reward': 50,
            'box_vel_reward': 20,
            'offset_reward': 1,
            'height_reward': 0.5,
            'upright_reward': 0.5,
            'alive_reward': 0.,
            'quat_reward': 0,
            'die_penalty': 10,
            'max_episode_steps': 200,
            'ant': 1,
            'direction': 'right',
            'init_randomness': 0.05,
            'diayn_reward': 0.1,
            "prob_perturb_action": 0.1,
            "perturb_action": 0.01,
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        # Env info
        self.ant = "ant_%d" % self._env_config['ant']
        self.box = "box_%d" % self._env_config['ant']
        self.ob_shape = OrderedDict([(self.ant, 41), (self.box, 6)])
        self.action_space.decompose(OrderedDict([(self.ant, 8)]))
        if self._env_config["direction"] == 'right':
            self.dx, self.dy = 1, 0
        if self._env_config["direction"] == 'left':
            self.dx, self.dy = -1, 0
        if self._env_config["direction"] == 'up':
            self.dx, self.dy = 0, 1
        if self._env_config["direction"] == 'down':
            self.dx, self.dy = 0, -1

    def _step(self, a):
        pos_before = self._get_pos('torso_geom')
        box_before = self._get_pos('box_geom')
        a = self._perturb_action(a)
        self.do_simulation(a)
        pos_after = self._get_pos('torso_geom')
        box_after = self._get_pos('box_geom')

        ob = self._get_obs()
        done = False
        vel_reward = 0
        height_reward = 0
        alive_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        height = self.data.qpos[2]
        vel = (pos_after[0] - pos_before[0]) * self.dx + \
            (pos_after[1] - pos_before[1]) * self.dy
        #offset = abs(pos_after[0] * self.dy) - abs(pos_before[0] * self.dy) + \
        #    abs(pos_after[1] * self.dx) - abs(pos_before[1] * self.dx)
        offset = abs((pos_after[0] - self._ant_pos[0]) * self.dy) + \
            abs((pos_after[1] - self._ant_pos[1]) * self.dx)

        box_vel = 0
        box_vel = (box_after[0] - box_before[0]) * self.dx + \
            (box_after[1] - box_before[1]) * self.dy

        box_forward = self._get_forward_vector('box_geom')
        quat_dist = cos_dist(self._box_forward, box_forward)

        box_dist = box_after - pos_after

        ant_quat = self._get_quat('torso')
        up = up_vector_from_quat(ant_quat)
        upright = cos_dist(up, np.array([0, 0, 1]))

        # reward
        vel_reward = self._env_config["vel_reward"] * vel
        box_vel_reward = self._env_config["box_vel_reward"] * box_vel
        offset_reward = self._env_config["offset_reward"] * (1 - min(2, offset))
        height_reward = -self._env_config["height_reward"] * np.abs(height - 0.6)
        alive_reward = self._env_config["alive_reward"]
        quat_reward = self._env_config["quat_reward"] * (quat_dist - 1)
        upright_reward = self._env_config["upright_reward"] * upright

        # fail
        done = not np.isfinite(self.data.qpos).all() or \
            0.35 > height or height > 0.9
        die_penalty = -self._env_config["die_penalty"] if done else 0
        #done = done or quat_dist < 0.9
        done = done or abs(box_dist[0]) > 5.5 or abs(box_dist[1]) > 5.5

        self._reward = reward = vel_reward + height_reward + \
            ctrl_reward + alive_reward + offset_reward + die_penalty + \
            quat_reward + upright_reward

        info = {"reward_vel": vel_reward,
                "reward_box_vel": box_vel_reward,
                "reward_offset": offset_reward,
                "reward_ctrl": ctrl_reward,
                "reward_height": height_reward,
                "reward_alive": alive_reward,
                "reward_quat": quat_reward,
                "reward_upright": upright_reward,
                "penalty_die": die_penalty,
                "box_forward": box_forward,
                "ant_pos": pos_after,
                "box_ob": ob[self.box],
                "success": self._success}

        return ob, reward, done, info

    def _get_obs(self):
        # ant
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        ant_pos = self._get_pos('torso')

        # box
        box_pos = self._get_pos('box_geom')
        #box_quat = self._get_quat('box')
        box_forward = self._get_forward_vector('box_geom')

        obs = OrderedDict([
            (self.ant, np.concatenate([qpos[2:15], qvel[:14], qacc[:14]])),
            (self.box, np.concatenate([box_pos - ant_pos, box_forward])),
            #('shared_pos', np.concatenate([qpos[2:7], qvel[:6], qacc[:6]])),
            #('lower_body', np.concatenate([qpos[7:15], qvel[6:14], qacc[6:14]])),
        ])

        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    @property
    def _init_qpos(self):
        # 3 for (x, y, z), 4 for (x, y, z, w), and 2 for each leg
        return np.array([0., 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         20., -1., 0.8, 1., 0., 0., 0.])

    @property
    def _init_qvel(self):
        return np.zeros(self.model.nv)

    def _reset(self):
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)
        #qpos[0:2] = 0
        qpos[2] = 0.58
        #qpos[3:7] = sample_quat()

        self.set_state(qpos, qvel)

        self._reset_ant_box()

        return self._get_obs()

    def _reset_ant_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # Initialize box
        init_box_pos = np.asarray([0, 0, 0.8])
        init_box_quat = sample_quat()
        #init_box_quat = [1, 0, 0, 0]
        qpos[15:18] = init_box_pos
        qpos[18:22] = init_box_quat

        self._box_forward = forward_vector_from_quat(init_box_quat)

        # Initialize ant
        x = np.random.uniform(low=-5.0, high=-3.0)
        y = np.random.uniform(low=-5.0, high=5.0)
        direction = self._env_config["direction"]
        if direction == "right":
            pass
        elif direction == "left":
            x = -x
        elif direction == "up":
            x, y = y, x
        elif direction == "down":
            x, y = y, -x

        qpos[0] = x
        qpos[1] = y
        self._ant_pos = (x, y)

        self.set_state(qpos, qvel)

