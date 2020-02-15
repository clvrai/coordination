from collections import OrderedDict

import numpy as np

from env.base import BaseEnv
from env.transform_utils import sample_quat

class TwoJacoEnv(BaseEnv):
    def __init__(self, xml_path, **kwargs):
        super().__init__(xml_path, **kwargs)

        # env info
        self.ob_shape = OrderedDict([("right_arm", 9 + 9), ("left_arm", 9 + 9), ("right_hand", 7 + 6), ("left_hand", 7 + 6), ("cube1", 7 + 6), ("cube2", 7 + 6)])
        self.action_space.decompose(OrderedDict([("right_arm", 9), ("left_arm", 9)]))

        # env config
        self._env_config.update({
            "init_pos_l": [0.3, 0.2, 0.86],
            "init_pos_r": [0.3, -0.2, 0.86],
            "init_radius": 0,
            "init_quat_randomness": 0, # np.pi / 6, appox. 0.5236
        })

    def _get_gripper_centers(self, update_markers=True):
        left_gripper_base = np.sum(np.array([self._get_pos("jaco_l_link_finger_{}".format(i + 1)) for i in range(3)]) * np.array([[0.5], [0.25], [0.25]]), 0)
        right_gripper_base = np.sum(np.array([self._get_pos("jaco_r_link_finger_{}".format(i + 1)) for i in range(3)]) * np.array([[0.5], [0.25], [0.25]]), 0)
        left_hand_pos = np.array(self._get_pos("jaco_l_link_hand"))
        right_hand_pos = np.array(self._get_pos("jaco_r_link_hand"))
        left_gripper_center = left_gripper_base - (left_hand_pos - left_gripper_base) * 0.75
        right_gripper_center = right_gripper_base - (right_hand_pos - right_gripper_base) * 0.75
        if update_markers:
            self._set_pos("marker1", left_gripper_center)
            self._set_pos("marker2", right_gripper_center)
        else:
            self._set_pos("marker1", [0, 0, 0])
            self._set_pos("marker2", [0, 0, 0])
        return [left_gripper_center, right_gripper_center]

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

        # cube1
        cube1_pos = self.data.get_joint_qpos("cube1")
        cube1_rel_pos = np.concatenate([cube1_pos[:3] - left_ref_pos, cube1_pos[3:]])
        cube1_vel = self.data.get_joint_qvel("cube1")

        # cube2
        cube2_pos = self.data.get_joint_qpos("cube2")
        cube2_rel_pos = np.concatenate([cube2_pos[:3] - right_ref_pos, cube2_pos[3:]])
        cube2_vel = self.data.get_joint_qvel("cube2")

        def clip(a):
            return np.clip(a, -30, 30)

        obs = OrderedDict([
            ('right_arm', np.concatenate([qpos[:9], clip(qvel[:9])])),
            ('left_arm', np.concatenate([qpos[9:18], clip(qvel[9:18])])),
            ('right_hand', np.concatenate([right_hand_pos, right_hand_quat,
                                           clip(right_hand_velp), clip(right_hand_velr)])),
            ('left_hand', np.concatenate([left_hand_pos, left_hand_quat,
                                          clip(left_hand_velp), clip(left_hand_velr)])),
            ('cube1', np.concatenate([cube1_rel_pos, clip(cube1_vel)])),
            ('cube2', np.concatenate([cube2_rel_pos, clip(cube2_vel)]))
        ])

        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    @property
    def _init_qpos(self):
        jaco_qpos = [-0.11, 0.03, 0.15, 0.38, -0.12, 0., 0., 0., 0.]
        return np.array(
                    jaco_qpos + jaco_qpos \
                    + [0] * 3 + [1] + [0] * 3 \
                    + (([0] * 3 + [1] + [0] * 3) if self.model.nq > 25 else []))

    @property
    def _init_qvel(self):
        return np.zeros(self.model.nv)

    def _random_init_pos(self, side='left'):
        base_x, base_y = self._env_config["init_pos_{}".format(side[0])][:2]
        dist = self._env_config['init_radius'] * np.sqrt(np.random.rand())
        ang = 2 * np.pi * np.random.rand()
        x = base_x + dist * np.cos(ang)
        y = base_y + dist * np.sin(ang)
        return x, y

    def reset_box(self):
        self.cube1_target_reached = False
        self.cube2_target_reached = False

        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # set box's initial position
        base_z_l = self._env_config["init_pos_l"][-1]
        x_box_pos, y_box_pos = self._random_init_pos(side='left')
        cube1_init_pos = [x_box_pos, y_box_pos, base_z_l]
        qpos[18:21] = np.asarray(cube1_init_pos)
        qpos[21:25] = sample_quat(low=-self._env_config['init_quat_randomness'],
                                  high=self._env_config['init_quat_randomness'])

        if len(qpos) > 25:
            base_z_r = self._env_config["init_pos_r"][-1]
            x_box_pos, y_box_pos = self._random_init_pos(side='right')
            cube2_init_pos = [x_box_pos, y_box_pos, base_z_r]
            qpos[25:28] = np.asarray(cube2_init_pos)
            qpos[28:32] = sample_quat(low=-self._env_config['init_quat_randomness'],
                                      high=self._env_config['init_quat_randomness'])

        self.set_state(qpos, qvel)

    def _reset(self):
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)
        self.set_state(qpos, qvel)

        self.reset_box()

        return self._get_obs()
