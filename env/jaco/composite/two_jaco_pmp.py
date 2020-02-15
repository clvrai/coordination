import os
import time
from collections import OrderedDict

import numpy as np

from env.jaco.two_jaco import TwoJacoEnv
from env.transform_utils import quat_dist, up_vector_from_quat, forward_vector_from_quat


def cos_dist(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


class TwoJacoPickMovePlaceEnv(TwoJacoEnv):
    def __init__(self, **kwargs):
        self.name = 'two-jaco-pick-move-place'
        super().__init__('two_jaco_pmp.xml', **kwargs)

        # config
        self._env_config.update({
            "train_left": True,
            "train_right": True,
            "success_reward": 100,
            "reach_reward": 10,
            "pick_reward": 30,
            "move_reward": 100,
            "place_reward": 100,
            "pick_height": 0.07,
            "move_dist": 0.15,
            "move_angle": 3.14,
            "init_randomness": 0.005,
            "ctrl_reward": 1e-4,
            "max_episode_steps": 100,
            "init_pos_l": [0.3, 0, 0.86],

            # randomness settings
            "init_radius": 0.005,
            "init_quat_randomness": 0,

            "diayn_reward": 0.01
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        # state
        self._hold_duration = [0, 0]
        self._box_z = [0, 0]
        self._t = 0
        self._pick_success = False
        self._place_success = False

        self._get_reference()

    def _get_reference(self):
        self.bar_body_id = self.sim.model.body_name2id("bar")
        self.bar_geom_id = self.sim.model.geom_name2id("bar")

    def _cos_vec(self, s, e1, e2):
        vec1 = np.array(e1) - s
        vec2 = np.array(e2) - s
        v1u = vec1 / np.linalg.norm(vec1)
        v2u = vec2 / np.linalg.norm(vec2)
        return np.clip(np.dot(v1u, v2u), -1.0, 1.0)

    def _compute_reward(self):
        ''' The goal is to pick, move, and place a long bar collaboratively.
            Reward has 4 components -- reach, pick, move, place.
        '''
        done = False
        on_ground = [False, False]
        off_table = [False, False]
        info = {}

        # compute gripper center positions
        gripper_center = self._get_gripper_centers()
        left_gripper_center, right_gripper_center = gripper_center

        # compute init gripper dist at first time step
        if self._t == 0:
            self._init_gripper_dists = [np.linalg.norm(gripper_center[i][::2] - self._init_bar_pos[::2]) for i in range(2)]
            self._init_bar_l_pos = self._get_pos('bar_l')
            self._init_bar_r_pos = self._get_pos('bar_r')

        # compute bar reaching reward
        # distance computed using x and z coordinates only
        # ie. horizontal holding position doesn't matter
        init_bar_pos = self._init_bar_pos
        lgpr_bar_dist = np.linalg.norm(left_gripper_center - self._get_pos('bar_l'))
        rgpr_bar_dist = np.linalg.norm(right_gripper_center - self._get_pos('bar_r'))
        reach_reward = -float(self._env_config['reach_reward'] * max(rgpr_bar_dist, lgpr_bar_dist))

        # compute bar pick-move-place reward
        init_bar_height = init_bar_pos[-1]
        bar_height = self._get_pos("bar")[-1]
        bar_height_l = self._get_pos("bar_l")[-1]
        bar_height_r = self._get_pos("bar_r")[-1]
        curr_bar_xy = self._get_pos('bar')[:2]
        curr_bar_l_xy = self._get_pos('bar_l')[:2]
        curr_bar_r_xy = self._get_pos('bar_r')[:2]
        final_bar_dxy = np.array([np.cos(self._env_config['move_angle']) * self._env_config['move_dist'],
                                  np.sin(self._env_config['move_angle']) * self._env_config['move_dist']])
        final_bar_xy = np.array(init_bar_pos[:2]) + final_bar_dxy
        final_bar_l_xy = np.array(self._init_bar_l_pos[:2]) + final_bar_dxy
        final_bar_r_xy = np.array(self._init_bar_r_pos[:2]) + final_bar_dxy
        
        # saved info
        pick_dist_l, pick_dist_r = 0, 0
        move_dist_l, move_dist_r = 0, 0
        place_dist_l, place_dist_r = 0, 0

        if not self._pick_success:
            # calculate pick reward
            target_bar_height = init_bar_height + self._env_config['pick_height']
            pick_dist_l = np.abs(bar_height_l - target_bar_height)
            pick_dist_r = np.abs(bar_height_r - target_bar_height)
            self._pick_reward = -float(self._env_config['pick_reward'] * max(pick_dist_l, pick_dist_r))

            # other rewards
            self._move_reward = -self._env_config['move_reward'] * 0.5
            self._place_reward = -self._env_config['place_reward'] * 0.5
            
            # check if pick is successful
            if bar_height_l - (init_bar_height + self._env_config['pick_height']) > -0.02 and \
                bar_height_r - (init_bar_height + self._env_config['pick_height']) > -0.02:
                self._pick_success = True

        if self._pick_success:
            # calculate move reward
            move_dist_l = np.linalg.norm(curr_bar_l_xy - final_bar_l_xy)
            move_dist_r = np.linalg.norm(curr_bar_r_xy - final_bar_r_xy)
            self._move_reward = -float(self._env_config['move_reward'] * max(move_dist_l, move_dist_r))

            # calculate place reward
            place_dist_l = np.abs(bar_height_l - init_bar_height) 
            place_dist_r = np.abs(bar_height_r - init_bar_height)
            self._place_reward = -float(self._env_config['place_reward'] * max(place_dist_l, place_dist_r))

            # check if place is successful
            if np.linalg.norm(curr_bar_l_xy - final_bar_l_xy) < 0.04 \
                      and np.linalg.norm(curr_bar_r_xy - final_bar_r_xy) < 0.04 \
                      and bar_height - init_bar_height < 0.02:
                self._place_success = True

        self._success = self._place_success

        done = self._success

        success_reward = 0
        if self._success:
            success_reward = self._env_config['success_reward']

        # sum rewards
        reward = reach_reward + \
                 self._pick_reward + self._place_reward + self._move_reward + \
                 success_reward

        info = {"reward_reach": reach_reward,
                "reward_pick": self._pick_reward,
                "reward_move": self._move_reward,
                "reward_place": self._place_reward,
                "reward_success": success_reward,
                "bar_pos": self._get_pos("bar"),
                "pick_success": self._pick_success,
                "place_success": self._place_success,
                "bar_pos_l" : self._get_pos("bar_l"),
                "bar_pos_r": self._get_pos("bar_r"),
                "pick_dist_l": pick_dist_l,
                "pick_dist_r": pick_dist_r,
                "move_dist_l": move_dist_l,
                "move_dist_r": move_dist_r,
                "place_dist_l": place_dist_l,
                "place_dist_r": place_dist_r,
                "success": self._success }

        return reward, done, info

    def _step(self, a):
        prev_reward, _, _ = self._compute_reward()

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
            dest_idx = new_dest_idx
        a = filled_action

        # scale actions from [-1, 1] range to actual control range
        mins = self.action_space.minimum
        maxs = self.action_space.maximum
        scaled_action = np.zeros_like(a)
        for i in range(self.action_space.size):
            scaled_action[i] = mins[i] + (maxs[i] - mins[i]) * (a[i] / 2 + 0.5)
        
        self.do_simulation(scaled_action)
        self._t += 1

        ob = self._get_obs()

        reward, done, info = self._compute_reward()
        ctrl_reward = self._ctrl_reward(scaled_action)
        info['reward_ctrl'] = ctrl_reward

        self._reward = reward - prev_reward + ctrl_reward

        return ob, self._reward, done, info

    def _get_obs(self):
        # jaco
        qpos = self.data.qpos
        qvel = self.data.qvel

        # reference position
        left_ref_pos = self._get_pos("jaco_l")
        right_ref_pos = self._get_pos("jaco_r")

        # left hand
        left_hand_name = "jaco_l_link_hand"
        left_hand_pos = self.data.get_body_xpos(left_hand_name) - left_ref_pos
        left_hand_quat = self.data.get_body_xquat(left_hand_name)
        left_hand_velp = self.data.get_body_xvelp(left_hand_name)
        left_hand_velr = self.data.get_body_xvelr(left_hand_name)

        # right hand
        right_hand_name = "jaco_r_link_hand"
        right_hand_pos = self.data.get_body_xpos(right_hand_name) - right_ref_pos
        right_hand_quat = self.data.get_body_xquat(right_hand_name)
        right_hand_velp = self.data.get_body_xvelp(right_hand_name)
        right_hand_velr = self.data.get_body_xvelr(right_hand_name)
        
        # bar
        bar_pos = self.data.get_joint_qpos("bar")
        bar_vel = self.data.get_joint_qvel("bar")

        # bar_l, bar_r
        bar_l_pos = self._get_pos("bar_l")
        bar_l_rel_pos = bar_l_pos - left_ref_pos
        bar_l_quat = self._get_quat("bar")
        bar_r_pos = self._get_pos("bar_r")
        bar_r_rel_pos = bar_r_pos - right_ref_pos
        bar_r_quat = self._get_quat("bar")

        def clip(a):
            return np.clip(a, -30, 30)

        obs = OrderedDict([
            ('right_arm', np.concatenate([qpos[:9], clip(qvel[:9])])),
            ('left_arm', np.concatenate([qpos[9:18], clip(qvel[9:18])])),
            ('right_hand', np.concatenate([right_hand_pos, right_hand_quat,
                                           clip(right_hand_velp), clip(right_hand_velr)])),
            ('left_hand', np.concatenate([left_hand_pos, left_hand_quat,
                                          clip(left_hand_velp), clip(left_hand_velr)])),
            ('bar', np.concatenate([bar_pos, clip(bar_vel)])),
            ('cube1', np.concatenate([bar_l_rel_pos, bar_l_quat, clip(bar_vel)])), # bar l
            ('cube2', np.concatenate([bar_r_rel_pos, bar_r_quat, clip(bar_vel)]))  # bar r
        ])

        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    def reset_box(self):
        self._t = 0
        self._pick_success = False
        self._place_success = False

        super().reset_box()

        self._pick_reward = 0
        self._move_reward = 0
        self._place_reward = 0
        self._hold_duration = [0, 0]
        self._bar_z = [0, 0]
        self._init_bar_pos = self._get_pos('bar')
        self._init_bar_quat = self._get_quat('bar')

