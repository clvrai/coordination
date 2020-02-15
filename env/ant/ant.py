from collections import OrderedDict

import numpy as np

from env.base import BaseEnv


class AntEnv(BaseEnv):
    def __init__(self, xml_path, **kwargs):
        super().__init__(xml_path, **kwargs)

        self._env_config.update({
            "init_randomness": 0.1,
            "init_random_rot": 0,
        })

        # Env info
        self.reward_type = ["ctrl_reward"]
        self.ob_shape = OrderedDict([("shared_pos", 17), ("lower_body", 24), ("upper_body", 9), ("box_pos", 3)])
        self.action_space.decompose(OrderedDict([("lower_body", 8), ("upper_body", 3)]))

    def _get_box_pos(self):
        return self._get_pos('box')

    def _render_callback(self):
        body_id = self.sim.model.body_name2id('torso')
        lookat = self.sim.data.body_xpos[body_id]
        lookat[2] += 0.3
        cam_pos = lookat + np.array([0, -10, 3])

        cam_id = self._camera_id
        self._set_camera_position(cam_id, cam_pos)
        self._set_camera_rotation(cam_id, lookat)

        self.sim.forward()

    def _viewer_reset(self):
        body_id = self.sim.model.body_name2id('torso')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self._viewer.cam.lookat[idx] = value

        self._viewer.cam.lookat[2] += 0.35
        self._viewer.cam.distance = 7
        #self._viewer.cam.azimuth = 180.
        self._viewer.cam.elevation = -25.
