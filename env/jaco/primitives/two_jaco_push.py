import os
import time
from collections import OrderedDict

import numpy as np

from env.jaco.two_jaco import TwoJacoEnv
from env.transform_utils import quat_dist


class TwoJacoPushEnv(TwoJacoEnv):
    def __init__(self, **kwargs):
        self.name = 'two-jaco-push'
        super().__init__('two_jaco_push.xml', **kwargs)

        # config
        self._env_config.update({
            "train_left": True,
            "train_right": True,
            "success_reward": 1000,
            "dist_reward": 100,
            "pos_reward": 500,
            "quat_reward": 30,
            "hold_duration": 10,
            "init_randomness": 0.005,
            "target_pos": [0.3, -0.03, 0.86],
            "ctrl_reward": 1e-4,
            "max_episode_steps": 100,
            "init_radius": 0.01
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        # state
        self._hold_duration = [0, 0]
        self._box_z = [0, 0]

        self._get_reference()

    def _get_reference(self):
        self.cube_body_id = [self.sim.model.body_name2id("cube{}".format(i)) for i in [1, 2]]
        self.cube_geom_id = [self.sim.model.geom_name2id("cube{}".format(i)) for i in [1, 2]]

    def _cos_vec(self, s, e1, e2):
        vec1 = np.array(e1) - s
        vec2 = np.array(e2) - s
        v1u = vec1 / np.linalg.norm(vec1)
        v2u = vec2 / np.linalg.norm(vec2)
        return np.clip(np.dot(v1u, v2u), -1.0, 1.0)

    def _compute_reward(self, before_step=False):
        done = False
        grasp_rewards = [0, 0]
        on_ground = [False, False]
        off_table = [False, False]
        info = {}

        # gripper center calculation
        left_gripper_center, right_gripper_center = self._get_gripper_centers()

        # dist reward
        cube1_pos = self._get_pos('cube1')
        cube2_pos = self._get_pos('cube2')
        dist_cube1 = np.linalg.norm(cube1_pos - left_gripper_center)
        dist_cube2 = np.linalg.norm(cube2_pos - right_gripper_center)
        if (not self.cube1_target_reached) and dist_cube1 < 0.1:
            self.cube1_target_reached = True
        if (not self.cube2_target_reached) and dist_cube2 < 0.1:
            self.cube2_target_reached = True
        dist_reward = [
            -(dist_cube1 if not self.cube1_target_reached else 0.05) * self._env_config["dist_reward"],
            -(dist_cube2 if not self.cube2_target_reached else 0.05) * self._env_config["dist_reward"]
        ]

        # pos reward
        target_pos = self._env_config['target_pos']
        target_pos_dist = [np.linalg.norm(cube1_pos - target_pos),
                       np.linalg.norm(cube2_pos - target_pos)]
        pos_reward = [-dist * self._env_config['pos_reward'] for dist in target_pos_dist]

        # quat reward
        target_quat_dist = [quat_dist(self._get_quat('cube{}'.format(i + 1)), self._init_box_quat[i]) for i in range(2)]
        quat_reward = [-dist * self._env_config['quat_reward'] for dist in target_quat_dist]

        # success
        success_reward = 0
        success_count = 0
        self._success = True
        early_stop = False
        if self._env_config["train_left"]:
            if target_pos_dist[0] < 0.02:
                if not before_step:
                    self._hold_duration[0] += 1
            elif self._hold_duration[0] > 0:
                early_stop = True
            self._success &= (self._hold_duration[0] > self._env_config['hold_duration'])
        if self._env_config["train_right"]:
            if target_pos_dist[1] < 0.02:
                if not before_step:
                    self._hold_duration[1] += 1
            elif self._hold_duration[1] > 0:
                early_stop = True
            self._success &= (self._hold_duration[1] > self._env_config['hold_duration'])

        if self._success:
            print('Reaching success!')
            success_reward = self._env_config["success_reward"]

        done = self._success or early_stop
        reward = success_reward
        if self._env_config["train_left"]:
            reward += pos_reward[0] + dist_reward[0] + quat_reward[0]
        if self._env_config["train_right"]:
            reward += pos_reward[1] + dist_reward[1] + quat_reward[1]

        info = {"reward_pos_1": pos_reward[0],
                "reward_pos_2": pos_reward[1],
                "reward_success": success_reward,
                "dist1": dist_cube1,
                "dist2": dist_cube2,
                "dist1_loss": dist_reward[0],
                "dist2_loss": dist_reward[1],
                "cube1_pos": self._get_pos("cube1"),
                "cube2_pos": self._get_pos("cube2"),
                "success": self._success }

        return reward, done, info

    def _step(self, a):
        prev_reward, _, _ = self._compute_reward(before_step=True)

        # build action from policy output
        filled_action = np.zeros((self.action_space.size,))
        inclusion_dict = { 'right_arm': self._env_config['train_right'],
                           'left_arm': self._env_config['train_left'] }
        dest_idx, src_idx = 0, 0
        for k in self.action_space.shape.keys():
            new_dest_idx = dest_idx + self.action_space.shape[k]
            if inclusion_dict[k]:
                new_src_idx = src_idx + self.action_space.shape[k]
                filled_action[dest_idx:new_dest_idx] = a[src_idx:new_src_idx]
                src_idx = new_src_idx
            else:
                # prevent non-moving arm from falling
                filled_action[dest_idx + 1] = -0.5 
            dest_idx = new_dest_idx
        a = filled_action
        
        # scale actions from [-1, 1] range to actual control range
        mins = self.action_space.minimum
        maxs = self.action_space.maximum
        scaled_action = np.zeros_like(a)
        for i in range(self.action_space.size):
            scaled_action[i] = mins[i] + (maxs[i] - mins[i]) * (a[i] / 2 + 0.5)

        self.do_simulation(scaled_action)

        ob = self._get_obs()

        reward, done, info = self._compute_reward()
        ctrl_reward = self._ctrl_reward(scaled_action)
        info['reward_ctrl'] = ctrl_reward

        self._reward = reward - prev_reward + ctrl_reward

        return ob, self._reward, done, info

    def reset_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # set box's initial position
        base_z_l = self._env_config["init_pos_l"][-1]
        x_box_pos, y_box_pos = self._random_init_pos(side='left')
        self._cube1_init_pos = [x_box_pos, y_box_pos, base_z_l]
        if not self._env_config['train_left']:
            self._cube1_init_pos = [x_box_pos, y_box_pos, 0]
        qpos[18:21] = np.asarray(self._cube1_init_pos)

        base_z_r = self._env_config["init_pos_r"][-1]
        x_box_pos, y_box_pos = self._random_init_pos(side='right')
        self._cube2_init_pos = [x_box_pos, y_box_pos, base_z_r]
        if not self._env_config['train_right']:
            self._cube2_init_pos = [x_box_pos, y_box_pos, 0]
        qpos[25:28] = np.asarray(self._cube2_init_pos)

        self.set_state(qpos, qvel)

        self.cube1_target_reached = False
        self.cube2_target_reached = False

        self._hold_duration = [0, 0]
        self._cube_z = [0, 0]

        cube1_init_quat = self._get_quat('cube1')
        cube2_init_quat = self._get_quat('cube2')
        self._init_box_quat = [cube1_init_quat, cube2_init_quat]
