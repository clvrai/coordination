import os
import time
from collections import OrderedDict

import numpy as np

from env.jaco.two_jaco import TwoJacoEnv
from env.transform_utils import quat_dist


class TwoJacoPickPushPlaceEnv(TwoJacoEnv):
    def __init__(self, **kwargs):
        self.name = 'two-jaco-pick-push-place'
        super().__init__('two_jaco_ppp.xml', **kwargs)

        # config
        self._env_config.update({
            "train_left": True,
            "train_right": True,
            "success_reward": 50,
            
            # dense reward
            "reach_reward": 10,
            "gripper_reward": 10,
            "pick_reward": 10,
            "push_reward": 30,
            "place_reward": 10,
            "ctrl_reward": 0,
            "reach_duration": 25,
            "pick_duration": 25,
            "place_duration": 25,
            "pick_height": 0.12,

            "init_randomness": 0.005,
            "max_episode_steps": 150,

            # randomness settings
            "init_radius": 0.0,
            "init_quat_randomness": 0,

            # diayn reward override
            "diayn_reward": 0.01
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        # state
        self._hold_duration = [0, 0]
        self._box_z = [0, 0]
        self._t = 0
        self._pick_success = False

        self._get_reference()

    def _get_reference(self):
        self.cube_body_id = [self.sim.model.body_name2id("cube{}".format(i)) for i in [1, 2]]
        self.cube_geom_id = [self.sim.model.geom_name2id("cube{}".format(i)) for i in [1, 2]]

        left_arm_geoms = ["jaco_l_link_1", "jaco_l_link_2", "jaco_l_link_3", "jaco_l_link_4", "jaco_l_link_5", "jaco_l_link_hand", "jaco_l_link_finger_1", "jaco_l_link_finger_2", "jaco_l_link_finger_3"]
        right_arm_geoms = ["jaco_r_link_1", "jaco_r_link_2", "jaco_r_link_3", "jaco_r_link_4", "jaco_r_link_5", "jaco_r_link_hand", "jaco_r_link_finger_1", "jaco_r_link_finger_2", "jaco_r_link_finger_3"]
        self.left_arm_geom_ids = [self.sim.model.geom_name2id(x) for x in left_arm_geoms]
        self.right_arm_geom_ids = [self.sim.model.geom_name2id(x) for x in right_arm_geoms]

        container_geom = "cube1_inside"
        bar_geom = "cube2"
        self.bar_geom_id = self.sim.model.geom_name2id(bar_geom)
        self.container_geom_id = self.sim.model.geom_name2id(container_geom)

    def _cos_vec(self, s, e1, e2):
        vec1 = np.array(e1) - s
        vec2 = np.array(e2) - s
        v1u = vec1 / np.linalg.norm(vec1)
        v2u = vec2 / np.linalg.norm(vec2)
        return np.clip(np.dot(v1u, v2u), -1.0, 1.0)

    def _compute_reward(self):
        ''' Cube 1 is the bar, cube 2 is the container.
            Reward has 4 components -- reach, pick, push, place.
            Push reward is based on distance to bar's xy position.
            Pick and place rewards are managed in two-staged manner.
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
            self._init_gripper_dists = [np.linalg.norm(gripper_center[i] - self._init_box_pos[i]) for i in range(2)]

        # compute criterias for success
        container_height = self._get_pos("cube1")[-1]
        bar_height = self._get_pos("cube2")[-1]

        # check container is on table
        container_on_table = np.abs(container_height - self._init_box_pos[0][-1]).item() < 0.003
        
        # check whether bar is in container
        bar_on_correct_height = np.abs(bar_height - self._init_box_pos[1][-1] - 0.02).item() < 0.003

        touch_bar_container = False
        lr_arm_contact = False
        for j in range(self.sim.data.ncon):
            c = self.sim.data.contact[j]
            if c.geom1 == self.bar_geom_id and c.geom2 == self.container_geom_id:
                touch_bar_container = True
            if c.geom1 == self.container_geom_id and c.geom2 == self.bar_geom_id:
                touch_bar_container = True
            if c.geom1 in self.left_arm_geom_ids and c.geom2 in self.right_arm_geom_ids:
                lr_arm_contact = True
            if c.geom1 in self.right_arm_geom_ids and c.geom2 in self.left_arm_geom_ids:
                lr_arm_contact = True

        bar_in_container = touch_bar_container # and bar_on_correct_height

        self._success = container_on_table and bar_in_container

        info = {
            "container_pos": np.round(self._get_pos("cube1"), 3),
            "bar_pos": np.round(self._get_pos("cube2"), 3),
            "container_on_table": container_on_table,
            "bar_on_correct_height": bar_on_correct_height,
            "bar_touches_container": touch_bar_container,
            "arm_contact": lr_arm_contact,
            "success": self._success,
        }

        # compute bar reaching reward
        init_bar_pos = self._init_box_pos[1]
        bar_dist = np.linalg.norm(right_gripper_center - init_bar_pos)
        target_dist = max(0.0, 1.0 - self._t / self._env_config['reach_duration']) * self._init_gripper_dists[0]
        reach_reward = -float(self._env_config['reach_reward'] * np.abs(bar_dist - target_dist))

        # gripper reward
        # calculates the cosine angle (cosang) between (l-finger - cube) and (r-finger - cube)
        # reward equals -(cosine angle) so that the gripper goes to two sides of the cube
        if self._t > self._env_config['reach_duration'] + self._env_config['pick_duration']:
            gripper_reward = self._gripper_reward
        else:
            cosang2 = np.mean([self._cos_vec(self._get_pos("cube2"),
                               self._get_pos('jaco_r_link_finger_{}'.format(i % 3 + 1)),
                               self._get_pos('jaco_r_link_finger_{}'.format((i + 1) % 3 + 1))) for i in range(3)]) \
                        * self._env_config["gripper_reward"]
            self._gripper_reward = gripper_reward = -cosang2
        gripper_reward *= self._env_config['gripper_reward']

        # compute bar pick-place reward
        init_bar_height = init_bar_pos[-1]
        bar_height = self._get_pos("cube2")[-1]
        reach_turnaround = self._env_config['reach_duration']
        pick_turnaround = reach_turnaround + self._env_config['pick_duration']
        place_turnaround = pick_turnaround + self._env_config['place_duration']
        target_pick_progress = max(0.0, min(1.0, (self._t - reach_turnaround) / pick_turnaround))
        target_place_remaining = max(0.0, min(1.0, (place_turnaround - self._t) / self._env_config['place_duration']))
        if target_pick_progress < 1:
            target_bar_height = target_pick_progress * self._env_config['pick_height'] + init_bar_height
            self._pick_reward = pick_reward = -float(self._env_config['pick_reward'] * np.abs(bar_height - target_bar_height))
            place_reward = 0
        else:
            target_bar_height = target_place_remaining * self._env_config['pick_height'] + init_bar_height
            pick_reward = self._pick_reward
            if self._pick_success:
                place_reward = -float(self._env_config['place_reward'] * np.abs(bar_height - target_bar_height))
            else:
                place_reward = -self._env_config['place_reward']

        # compute container push reward
        init_container_pos = np.array(self._init_box_pos[0][:2])
        curr_container_pos = np.array(self._get_pos("cube1")[:2])
        curr_bar_pos = self._get_pos("cube2")[:2]
        target_push_progress = max(0.0, min(1.0, (self._t - reach_turnaround) / place_turnaround))
        target_container_pos = curr_bar_pos * target_push_progress + init_container_pos * (1.0 - target_push_progress)
        push_reward = -float(self._env_config['push_reward'] * np.linalg.norm(curr_container_pos - target_container_pos))

        # compute success reward
        success_reward = 0
        if self._success:
            success_reward = self._env_config['success_reward'] 

        # sum rewards
        reward = min(reach_reward + gripper_reward + pick_reward + place_reward, push_reward) + success_reward

        info.update({
            "reward_reach": reach_reward,
            "reward_gripper": gripper_reward,
            "reward_pick": pick_reward,
            "reward_place": place_reward,
            "reward_push": push_reward,
            "reward_success": success_reward,
            "reward": reward
        })

        done = self._success

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
        ctrl_reward = self._ctrl_reward(scaled_action) * self._env_config['ctrl_reward']
        info['reward_ctrl'] = ctrl_reward

        self._reward = reward - prev_reward + ctrl_reward

        return ob, self._reward, done, info

    def reset_box(self):
        self._t = 0
        self._pick_success = False

        self.cube1_target_reached = False
        self.cube2_target_reached = False

        super().reset_box()

        self._hold_duration = [0, 0]
        self._cube_z = [0, 0]
        cube1_init_pos = self._get_pos('cube1')
        cube2_init_pos = self._get_pos('cube2')
        self._init_box_pos = [cube1_init_pos, cube2_init_pos]

        cube1_init_quat = self._get_quat('cube1')
        cube2_init_quat = self._get_quat('cube2')
        self._init_box_quat = [cube1_init_quat, cube2_init_quat]
