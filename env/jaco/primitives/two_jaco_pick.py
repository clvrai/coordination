import os
import time
from collections import OrderedDict

import numpy as np

from env.jaco.two_jaco import TwoJacoEnv
from env.transform_utils import quat_dist, up_vector_from_quat, forward_vector_from_quat


def cos_dist(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


class TwoJacoPickEnv(TwoJacoEnv):
    def __init__(self, **kwargs):
        self.name = 'two-jaco-pick'
        super().__init__('two_jaco_pick.xml', **kwargs)

        # config
        self._env_config.update({
            "train_left": True,
            "train_right": True,
            "prob_perturb_action": 0.0,
            "perturb_action": 0.6,
            "success_reward": 100,
            "success_reward_penalty": 10,
            "pick_reward": 500,
            "pick_height": 0.2,
            "dist_reward": 100,
            "ang_reward": 300, # angle of hand, max_angle = 1
            "grasp_reward": 50,
            "in_hand_reward": 0,
            "pos_reward": 1000, # dist of box
            "quat_reward": 1000, # angle of box
            "accumulate_pos_quat_reward": False, # whether to add absolute reward each time or the offset
            "gripper_reward": 0,
            "hold_duration": 20,
            "hold_reward": 10,
            "init_randomness": 0.005,
            "init_quat_randomness": 0.05236,
            "check_contact_for_in_air": False,
            "in_air_threshold": 0.02,
            "ctrl_reward": 1e-4,
            "unstable_threshold": 0.2,
            "max_episode_steps": 100,
            "init_radius": 0.015,
            "diayn_reward": 0.01,
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

    def _compute_reward(self, count_hold=False):
        done = False
        on_ground = [False, False]
        off_table = [False, False]
        info = {}

        cube1_pos = self._get_pos("cube1")
        cube2_pos = self._get_pos("cube2")

        cube_z = [cube1_pos[-1], cube2_pos[-1]]
        left_gripper_center, right_gripper_center = self._get_gripper_centers()

        # dist reward
        cube1_target = cube1_pos
        cube2_target = cube2_pos
        dist_cube1 = np.linalg.norm(cube1_target - left_gripper_center)
        dist_cube2 = np.linalg.norm(cube2_target - right_gripper_center)
        if (not self.cube1_target_reached) and dist_cube1 < 0.04:
            self.cube1_target_reached = True
        if (not self.cube2_target_reached) and dist_cube2 < 0.04:
            self.cube2_target_reached = True
        dist_reward = [
            -(dist_cube1 if not self.cube1_target_reached else 0.) * self._env_config["dist_reward"],
            -(dist_cube2 if not self.cube2_target_reached else 0.) * self._env_config["dist_reward"]
        ]

        dist_hand = [np.linalg.norm(cube1_pos - left_gripper_center),
                     np.linalg.norm(cube2_pos - right_gripper_center)]

        # gripper reward
        # calculates the cosine angle (cosang) between (l-finger - cube) and (r-finger - cube)
        # reward equals -(cosine angle) so that the gripper goes to two sides of the cube
        cosang1 = np.mean([self._cos_vec(cube1_pos,
                           self._get_pos('jaco_l_link_finger_{}'.format(i % 3 + 1)),
                           self._get_pos('jaco_l_link_finger_{}'.format((i + 1) % 3 + 1))) for i in range(3)]) \
                    * self._env_config["gripper_reward"]
        cosang2 = np.mean([self._cos_vec(cube2_pos,
                           self._get_pos('jaco_r_link_finger_{}'.format(i % 3 + 1)),
                           self._get_pos('jaco_r_link_finger_{}'.format((i + 1) % 3 + 1))) for i in range(3)]) \
                    * self._env_config["gripper_reward"]
        gripper_reward = [-cosang1, -cosang2]

        self._success = True
        in_hand_rewards = [0, 0]
        pick_rewards = [0, 0]
        grasp_count = [0, 0]
        grasp_rewards = [0, 0]
        hold_rewards = [0, 0]
        pick_unstable = [False, False]
        in_hands = [False, False]
        in_air = [False, False]
        base_z = self._env_config['init_pos_l'][-1]
        for i in range(2):
            # check whether box is in air by checking contact
            if self._env_config['check_contact_for_in_air']:
                for j in range(self.sim.data.ncon):
                    c = self.sim.data.contact[j]
                    cube_geom_id = self.sim.model.geom_name2id('cube{}'.format(i + 1))
                    table_geom_id = self.sim.model.geom_name2id('table_collision')
                    if c.geom1 == cube_geom_id and c.geom2 == table_geom_id:
                        in_air[i] = True
                    if c.geom2 == cube_geom_id and c.geom1 == table_geom_id:
                        in_air[i] = True
            else:
                in_air[i] = cube_z[i] >= base_z \
                                + self._env_config["in_air_threshold"]

            on_ground[i] = not in_air[i]
            in_hands[i] = dist_hand[i] < 0.06
            off_table[i] = cube_z[i] < 0.8
            self._cube_z[i] = cube_z[i]
            pick_height = cube_z[i] - base_z

            count_success = (self._env_config['train_left'] and i == 0) or \
                            (self._env_config['train_right'] and i == 1)

            # in hand reward
            in_hand_rewards[i] = self._env_config['in_hand_reward'] if in_hands[i] else 0

            # pick reward
            if in_hands[i] and in_air[i]:
                if not count_success: pick_height = 0
                capped_pick_height = min(pick_height, self._env_config['pick_height'])
                pick_rewards[i] = self._env_config["pick_reward"] * capped_pick_height

                if count_hold and pick_height > 0.12:
                    self._hold_duration[i] += 1
                if self._hold_duration[i] >= self._env_config['hold_duration']:
                    print('Cube {}: success pick!'.format(i + 1))
                elif count_success:
                    self._success = False
            elif count_success:
                self._success = False

            # grasp reward
            contact_flag = [False] * 3
            geom_name_prefix = 'jaco_{}_link_finger'.format('l' if i == 0 else 'r')
            for j in range(self.sim.data.ncon):
                c = self.sim.data.contact[j]
                for k in range(3):
                    geom_name = '{}_{}'.format(geom_name_prefix, k + 1)
                    geom_id = self.sim.model.geom_name2id(geom_name)
                    if c.geom1 == geom_id and c.geom2 == self.cube_geom_id[i]:
                        contact_flag[k] = True
                    if c.geom2 == geom_id and c.geom1 == self.cube_geom_id[i]:
                        contact_flag[k] = True
            grasp_count[i] = np.array(contact_flag).astype(int).sum()
            if grasp_count[i] == 3:
                grasp_rewards[i] = self._env_config["grasp_reward"]

            # check pick stable
            xy_offset = np.linalg.norm(np.array(self._get_pos('cube{}'.format(i + 1)))[:2] - self._init_box_pos[i][:2])
            if xy_offset > self._env_config['unstable_threshold']:
                pick_unstable[i] = True

            # hold reward
            hold_rewards[i] = self._hold_duration[i] * self._env_config['hold_reward']

        # ang reward
        ang_lhand = cos_dist(
            left_gripper_center - self.data.get_body_xpos("jaco_l_link_hand"), np.array([0, 0, -1]))
        ang_rhand = cos_dist(
            right_gripper_center - self.data.get_body_xpos("jaco_r_link_hand"), np.array([0, 0, -1]))

        ang_reward = [
            ang_lhand * self._env_config["ang_reward"],
            ang_rhand * self._env_config["ang_reward"]
        ]

        done = self._success
        if self._env_config["train_left"]:
            done |= off_table[0] or pick_unstable[0]
            # done |= (self._hold_duration[0] > 0 and on_ground[0])
        if self._env_config["train_right"]:
            done |= off_table[1] or pick_unstable[1]
            # done |= (self._hold_duration[1] > 0 and on_ground[1])

        reward = 0
        if self._env_config["train_left"]:
            reward += pick_rewards[0] + in_hand_rewards[0] \
                    + ang_reward[0] + gripper_reward[0] \
                    + dist_reward[0] + hold_rewards[0]
        if self._env_config["train_right"]:
            reward += pick_rewards[1] + in_hand_rewards[1] \
                    + ang_reward[1] + gripper_reward[1] \
                    + dist_reward[1] + hold_rewards[1]

        def organize_info(l):
            # only show info of trained arms
            if self._env_config['train_left'] and self._env_config['train_right']:
                return np.round(l, 4)
            elif self._env_config['train_left']:
                return np.round(l[0], 4)
            elif self._env_config['train_right']:
                return np.round(l[1], 4)

        info = {"reward_dist": organize_info(dist_reward),
                "reward_pick": organize_info(pick_rewards),
                "reward_hold": organize_info(hold_rewards),
                "reward_ang": organize_info(ang_reward),
                "reward_gripper": organize_info(gripper_reward),
                "reward_grasp": organize_info(grasp_rewards),
                "reward_in_hand": organize_info(in_hand_rewards),
                "dist": organize_info([dist_cube1, dist_cube2]),
                "in_hands_1": in_hands[0],
                "in_hands_2": in_hands[1],
                "in_air_1": in_air[0],
                "in_air_2": in_air[1],
                "hold_duration": organize_info(self._hold_duration),
                "grasp_count": organize_info(grasp_count),
                "cube_pos": organize_info([cube1_pos, cube2_pos]),
                "cube_target": organize_info([cube1_target, cube2_target]),
                "gripper_pos": organize_info([left_gripper_center, right_gripper_center]),
                "success": self._success }

        return reward, done, info

    def _step(self, a):
        prev_reward, _, old_info = self._compute_reward()

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

        # scale actions from [-1, 1] range to actual control range
        mins = self.action_space.minimum
        maxs = self.action_space.maximum
        scaled_action = mins + (maxs - mins) * (filled_action / 2 + 0.5)

        # perturb action
        if np.random.rand() < self._env_config['prob_perturb_action']:
            perturb_unit = np.array(maxs - mins) / 2 * self._env_config['perturb_action']
            perturb = (np.random.rand(len(mins)) * 2 - 1) * perturb_unit
        else:
            perturb = np.zeros(len(mins))
        scaled_action += perturb

        # maybe apply external force
        if np.random.rand() < self._env_config['prob_apply_force']:
            self._apply_external_force()
            force = True
        else:
            force = False

        # do simulation
        self.do_simulation(scaled_action)

        # reset applied force
        self._reset_external_force()

        ob = self._get_obs()

        reward, done, info = self._compute_reward(count_hold=True)

        for k in info:
            if k.startswith('reward_'):
                info[k] = info[k] - old_info[k]

        ctrl_reward = self._ctrl_reward(scaled_action)
        info['reward_ctrl'] = ctrl_reward
        info['perturb_ac'] = perturb
        info['ext_force'] = force is not None

        # stable reward
        stable_rewards = [0, 0]
        for i in range(2):
            if self._env_config['accumulate_pos_quat_reward']:
                reference_box_pos = self._init_box_pos
                reference_box_quat = self._init_box_quat
            else:
                reference_box_pos = self._prev_box_pos
                reference_box_quat = self._prev_box_quat

            target_pos_dist = np.linalg.norm(self._get_pos('cube{}'.format(i+1))[:2] - reference_box_pos[i][:2])
            target_quat_dist = cos_dist(up_vector_from_quat(self._get_quat('cube{}'.format(i+1))),
                                        up_vector_from_quat(reference_box_quat[i])) + \
                cos_dist(forward_vector_from_quat(self._get_quat('cube{}'.format(i+1))),
                         forward_vector_from_quat(reference_box_quat[i])) - 2

            self._prev_box_pos = [self._get_pos('cube{}'.format(i + 1)) for i in range(2)]
            self._prev_box_quat = [self._get_quat('cube{}'.format(i + 1)) for i in range(2)]

            if info['in_hands_{}'.format(i+1)] and info['in_air_{}'.format(i+1)]:
                stable_rewards[i] = -self._env_config["pos_reward"] * target_pos_dist \
                                    + self._env_config["quat_reward"] * target_quat_dist
            else:
                stable_rewards[i] = 0
            info['reward_pos_{}'.format(i+1)] = -self._env_config["pos_reward"] * target_pos_dist
            info['reward_quat_{}'.format(i+1)] = self._env_config["quat_reward"] * target_quat_dist
            info['cube_up_{}'.format(i+1)] = up_vector_from_quat(self._get_quat('cube{}'.format(i+1)))
            info['cube_init_up_{}'.format(i+1)] = up_vector_from_quat(self._init_box_quat[i])

        total_stable_reward = 0
        if self._env_config['train_left']:
            total_stable_reward += stable_rewards[0]
        if self._env_config['train_right']:
            total_stable_reward += stable_rewards[1]

        success_reward = 0
        if self._success:
            print('All picks success!')
            success_reward = self._env_config["success_reward"]
        info['reward_success'] = success_reward

        self._reward = reward - prev_reward + ctrl_reward + \
            success_reward + total_stable_reward

        return ob, self._reward, done, info

    def reset_box(self):
        super().reset_box()

        self.cube1_target_reached = False
        self.cube2_target_reached = False

        self._hold_duration = [0, 0]
        self._cube_z = [0, 0]
        cube1_init_pos = self._get_pos('cube1')
        cube2_init_pos = self._get_pos('cube2')
        self._init_box_pos = [cube1_init_pos, cube2_init_pos]

        cube1_init_quat = self._get_quat('cube1')
        cube2_init_quat = self._get_quat('cube2')
        self._init_box_quat = [cube1_init_quat, cube2_init_quat]

        self._prev_box_pos = self._init_box_pos
        self._prev_box_quat = self._init_box_quat
