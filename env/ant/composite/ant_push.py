from collections import OrderedDict

import numpy as np

from env.ant.ant import AntEnv
from env.transform_utils import up_vector_from_quat, forward_vector_from_quat, \
    l2_dist, cos_dist, sample_quat


class AntPushEnv(AntEnv):
    def __init__(self, **kwargs):
        self.name = 'ant_push'
        super().__init__('ant_push.xml', **kwargs)

        # Env info
        #self.ob_shape = OrderedDict([("ant_1_shared_pos", 19), ("ant_1_lower_body", 24), ("ant_2_shared_pos", 19), ("ant_2_lower_body", 24), ("box_pos", 3)])
        self.ob_shape = OrderedDict([("ant_1", 41), ("ant_2", 41),
                                     ("box_1", 6), ("box_2", 6),
                                     ("goal_1", 3), ("goal_2", 3)])
        self.action_space.decompose(OrderedDict([("ant_1", 8), ("ant_2", 8)]))

        # Env config
        self._env_config.update({
            'random_ant_pos': 0.01,
            'random_goal_pos': 0.01,
            #'random_goal_pos': 0.5,
            'dist_threshold': 0.5,
            'quat_threshold': 0.9,
            'ant_dist_reward': 50,
            'goal_dist_reward': 200,
            'goal1_dist_reward': 2,
            'goal2_dist_reward': 2,
            'quat_reward': 10, # quat_dist usually between 0.95 ~ 1
            'height_reward': 0.5,
            'upright_reward': 0.5,
            'alive_reward': 0.,
            'success_reward': 500,
            'die_penalty': 10,
            'sparse_reward': 0,
            'init_randomness': 0.01,
            #'max_episode_steps': 400,
            'max_episode_steps': 200,
        }) 
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        self._ant1_push = False
        self._ant2_push = False
        self._success_count = 0

    def _step(self, a):
        ant1_pos_before = self._get_pos('ant_1_torso_geom')
        ant2_pos_before = self._get_pos('ant_2_torso_geom')
        box1_pos_before = self._get_pos('box_geom1')
        box2_pos_before = self._get_pos('box_geom2')
        box_pos_before = self._get_pos('box')
        box_quat_before = self._get_quat('box')

        self.do_simulation(a)

        ant1_pos = self._get_pos('ant_1_torso_geom')
        ant2_pos = self._get_pos('ant_2_torso_geom')
        box1_pos = self._get_pos('box_geom1')
        box2_pos = self._get_pos('box_geom2')
        box_pos = self._get_pos('box')
        box_quat = self._get_quat('box')
        goal1_pos = self._get_pos('goal_geom1')
        goal2_pos = self._get_pos('goal_geom2')
        goal_pos = self._get_pos('goal')
        goal_quat = self._get_quat('goal')

        ob = self._get_obs()
        done = False
        ctrl_reward = self._ctrl_reward(a)

        goal_forward = forward_vector_from_quat(goal_quat)
        box_forward = forward_vector_from_quat(box_quat)
        box_forward_before = forward_vector_from_quat(box_quat_before)

        goal1_dist = l2_dist(goal1_pos, box1_pos)
        goal1_dist_before = l2_dist(goal1_pos, box1_pos_before)
        goal2_dist = l2_dist(goal2_pos, box2_pos)
        goal2_dist_before = l2_dist(goal2_pos, box2_pos_before)
        goal_dist = l2_dist(goal_pos, box_pos)
        goal_dist_before = l2_dist(goal_pos, box_pos_before)
        goal_quat = cos_dist(goal_forward, box_forward)
        goal_quat_before = cos_dist(goal_forward, box_forward_before)

        ant1_height = self.data.qpos[2]
        ant2_height = self.data.qpos[17]
        ant1_quat = self._get_quat('ant_1_torso')
        ant1_up = up_vector_from_quat(ant1_quat)
        ant1_upright = cos_dist(ant1_up, np.array([0, 0, 1]))
        ant2_quat = self._get_quat('ant_2_torso')
        ant2_up = up_vector_from_quat(ant2_quat)
        ant2_upright = cos_dist(ant2_up, np.array([0, 0, 1]))

        ant1_dist = l2_dist(ant1_pos_before[:2], box1_pos_before[:2]) - l2_dist(ant1_pos[:2], box1_pos[:2])
        ant2_dist = l2_dist(ant2_pos_before[:2], box2_pos_before[:2]) - l2_dist(ant2_pos[:2], box2_pos[:2])

        if l2_dist(ant1_pos[:2], box1_pos[:2]) < 2.0: self._ant1_push = True
        if l2_dist(ant2_pos[:2], box2_pos[:2]) < 2.0: self._ant2_push = True

        success_reward = 0
        ant1_dist_reward = self._env_config["ant_dist_reward"] * ant1_dist if not self._ant1_push else 0
        ant2_dist_reward = self._env_config["ant_dist_reward"] * ant2_dist if not self._ant2_push else 0
        #goal1_dist_reward = self._env_config["goal1_dist_reward"] * (goal1_dist_before - goal1_dist)
        #goal2_dist_reward = self._env_config["goal2_dist_reward"] * (goal2_dist_before - goal2_dist)
        goal1_dist_reward = self._env_config["goal1_dist_reward"] * 1 if goal1_dist < self._env_config["dist_threshold"] else 0
        goal2_dist_reward = self._env_config["goal2_dist_reward"] * 1 if goal2_dist < self._env_config["dist_threshold"] else 0
        goal_dist_reward = self._env_config["goal_dist_reward"] * (goal_dist_before - goal_dist)
        quat_reward = -self._env_config["quat_reward"] * (1 - goal_quat)
        #quat_reward = self._env_config["quat_reward"] * (goal_quat - goal_quat_before) if goal_pos[0] >= box_pos[0] else 0
        alive_reward = self._env_config["alive_reward"]
        ant1_height_reward = -self._env_config["height_reward"] * np.abs(ant1_height - 0.6)
        ant2_height_reward = -self._env_config["height_reward"] * np.abs(ant2_height - 0.6)
        ant1_upright_reward = self._env_config["upright_reward"] * ant1_upright
        ant2_upright_reward = self._env_config["upright_reward"] * ant2_upright

        if goal1_dist < self._env_config["dist_threshold"] and \
                goal2_dist < self._env_config["dist_threshold"] and \
                goal_quat > self._env_config["quat_threshold"]:
            self._success_count += 1
            success_reward = 10
            if self._success_count >= 1:
                self._success = True
                success_reward = self._env_config["success_reward"]

        # fail
        done = not np.isfinite(self.data.qpos).all() or \
            0.35 > ant1_height or ant1_height > 0.9 or \
            0.35 > ant2_height or ant2_height > 0.9
        #done = done or abs(ant1_pos[0]) > 7 or abs(ant2_pos[0]) > 7
        #done = done or abs(ant1_pos[1]) > 6 or abs(ant2_pos[1]) > 6
        die_penalty = -self._env_config["die_penalty"] if done else 0
        done = done or self._success

        if self._env_config['sparse_reward']:
            self._reward = reward = self._success == 1
        else:
            self._reward = reward = ant1_dist_reward + ant2_dist_reward + \
                ant1_height_reward + ant2_height_reward + ant1_upright_reward + ant2_upright_reward + \
                goal_dist_reward + goal1_dist_reward + goal2_dist_reward + quat_reward + \
                alive_reward + success_reward + ctrl_reward + die_penalty

        info = {"success": self._success,
                "reward_ant1_dist": ant1_dist_reward,
                "reward_ant2_dist": ant2_dist_reward,
                "reward_ant1_height": ant1_height_reward,
                "reward_ant2_height": ant2_height_reward,
                "reward_ant1_upright": ant1_upright_reward,
                "reward_ant2_upright": ant2_upright_reward,
                "reward_goal_dist": goal_dist_reward,
                "reward_goal1_dist": goal1_dist_reward,
                "reward_goal2_dist": goal2_dist_reward,
                "reward_quat": quat_reward,
                "reward_alive": alive_reward,
                "reward_ctrl": ctrl_reward,
                "die_penalty": die_penalty,
                "reward_success": success_reward,
                "ant1_pos": ant1_pos,
                "ant2_pos": ant2_pos,
                "ant1_up": ant1_up,
                "ant2_up": ant2_up,
                "goal_quat": goal_quat,
                "goal_dist": goal_dist,
                "goal1_dist": goal1_dist,
                "goal2_dist": goal2_dist,
                "box1_ob": ob['box_1'],
                "box2_ob": ob['box_2'],
                "goal1_ob": ob['goal_1'],
                "goal2_ob": ob['goal_2'],
                }

        return ob, reward, done, info

    def _get_obs(self):
        # ant
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        ant_pos1 = self._get_pos('ant_1_torso')
        ant_pos2 = self._get_pos('ant_2_torso')

        # box
        box_pos1 = self._get_pos('box_geom1')
        box_pos2 = self._get_pos('box_geom2')
        #box_quat1 = self._get_quat('box')
        #box_quat2 = self._get_quat('box')
        box_forward1 = self._get_forward_vector('box_geom1')
        box_forward2 = self._get_forward_vector('box_geom2')

        # goal
        goal_pos1 = self._get_pos('goal_geom1')
        goal_pos2 = self._get_pos('goal_geom2')

        obs = OrderedDict([
            #('ant_1_shared_pos', np.concatenate([qpos[:7], qvel[:6], qacc[:6]])),
            #('ant_1_lower_body', np.concatenate([qpos[7:15], qvel[6:14], qacc[6:14]])),
            #('ant_2_shared_pos', np.concatenate([qpos[15:22], qvel[14:22], qacc[14:22]])),
            #('ant_2_lower_body', np.concatenate([qpos[22:30], qvel[22:30], qacc[22:30]])),
            ('ant_1', np.concatenate([qpos[2:15], qvel[:14], qacc[:14]])),
            ('ant_2', np.concatenate([qpos[17:30], qvel[14:28], qacc[14:28]])),
            ('box_1', np.concatenate([box_pos1 - ant_pos1, box_forward1])),
            ('box_2', np.concatenate([box_pos2 - ant_pos2, box_forward2])),
            ('goal_1', goal_pos1 - box_pos1),
            ('goal_2', goal_pos2 - box_pos2),
        ])

        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    @property
    def _init_qpos(self):
        # 3 for (x, y, z), 4 for (x, y, z, w), and 2 for each leg
        return np.array([0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0., 0., 0.8, 1., 0., 0., 0.])

    @property
    def _init_qvel(self):
        return np.zeros(self.model.nv)

    def _reset(self):
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)

        qpos[2] = 0.58
        qpos[17] = 0.58

        self.set_state(qpos, qvel)

        self._reset_ant_box()

        self._ant1_push = False
        self._ant2_push = False
        self._success_count = 0

        return self._get_obs()

    def _reset_ant_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # Initialize box
        init_box_pos = np.asarray([0, 0, 0.8])
        #init_box_quat = sample_quat(low=-np.pi/32, high=np.pi/32)
        init_box_quat = sample_quat(low=0, high=0)
        qpos[30:33] = init_box_pos
        qpos[33:37] = init_box_quat

        # Initialize ant
        #qpos[0:2] = [-4, 2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        #qpos[15:17] = [-4, -2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        qpos[0:2] = [-2, 2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        qpos[15:17] = [-2, -2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        #qpos[0:2] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], 2]
        #qpos[15:17] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], -2]

        # Initialize goal
        #x = 4.5 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        x = 3 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        y = 0 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        z = 0.8
        goal_pos = np.asarray([x, y, z])
        #goal_quat = sample_quat()
        self._set_pos('goal', goal_pos)
        #self._set_quat('goal', goal_quat)

        self.set_state(qpos, qvel)

    def _render_callback(self):
        lookat = [0.5, 0, 0]
        cam_pos = lookat + np.array([-4.5, -12, 10])

        cam_id = self._camera_id
        self._set_camera_position(cam_id, cam_pos)
        self._set_camera_rotation(cam_id, lookat)

        self.sim.forward()

