import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDrawerPutBlockEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        drawer_obj_low = (-0.1, 0.9, 0.0)
        drawer_obj_high = (0.1, 0.9, 0.0)
        obj_low = (-0.15, 0.4, 0.02)
        obj_high = (0.15, 0.5, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "obj_init_angle": np.array(
                [
                    0.3,
                ],
                dtype=np.float32,
            ),
            "obj_init_pos": np.array([0.0, 0.4, 0.02], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.hstack((drawer_obj_low, obj_low)),
            np.hstack((drawer_obj_high, obj_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer_put_block.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        success = self.compute_reward(action, obs)
        # print(obj_to_target)
        # print(float(obj_to_target <= 0.03))
        info = {"success": success}
        return 0.0, info
        # (
        #     reward,
        #     gripper_error,
        #     gripped,
        #     handle_error,
        #     caging_reward,
        #     opening_reward,
        # ) = self.compute_reward(action, obs)

        # info = {
        #     "success": float(handle_error <= 0.03),
        #     "near_object": float(gripper_error <= 0.03),
        #     "grasp_success": float(gripped > 0),
        #     "grasp_reward": caging_reward,
        #     "in_place_reward": opening_reward,
        #     "obj_to_target": handle_error,
        #     "unscaled_reward": reward,
        # }

        # return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("objGeom")

    def _get_pos_objects(self):
        drawer_handle_pos = self.get_body_com("drawer_link") + np.array(
            [0.0, -0.16, 0.0]
        )
        block_pos = self.get_body_com("obj")
        return np.concatenate([drawer_handle_pos, block_pos])

    def _get_quat_objects(self):
        drawer_quat = self.sim.data.get_body_xquat("drawer_link")
        block_quat = self.sim.data.get_body_xquat("obj")
        return np.concatenate([drawer_quat, block_quat])

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos = (
            self._get_state_rand_vec()
            if self.random_init
            else self.init_config["obj_init_pos"]
        )

        self.drawer_init_pos = self.obj_init_pos[:3]
        self.block_init_pos = self.obj_init_pos[3:]
        # also randomize block position

        # print("drawer")
        # print(self.drawer_init_pos)
        # print("block")
        # print(self.block_init_pos)

        # Set mujoco body to computed position
        self.sim.model.body_pos[
            self.model.body_name2id("drawer")
        ] = self.drawer_init_pos

        # print(self.model.body_name2id("drawer"))
        # print(self.model.body_name2id("obj"))
        self.sim.model.body_pos[self.model.body_name2id("obj")] = self.block_init_pos
        self._set_obj_xyz(self.fix_extreme_obj_pos(self.block_init_pos))

        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos_drawer = self.drawer_init_pos + np.array(
            [0.0, -0.16 - self.maxDist, 0.09]
        )
        self._target_pos = self._target_pos_drawer
        # target should be a little below initial drawer position
        # self._target_pos = self._target_pos_drawer + np.array([0, -0.04, -0.02])

        # base_drawer_pos = goal_pos - np.array([0, 0, 0.3])
        # self.sim.model.body_pos[self.model.body_name2id("drawer")] = base_shelf_pos[-3:]
        # self._target_pos_block = self.sim.model.body_pos[
        #     self.model.body_name2id("drawer")
        # ]

        return self._get_obs()

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        obj = obs[11:14]
        # target = self._target_pos
        # obj_to_target = np.linalg.norm(obj - target)

        # success is if the obj is within a certain radius of center of drawer
        success = False

        less_than_z = bool(obj[2] < 0.08)
        within_box_x = bool(abs(obj[0] - self.drawer_init_pos[0]) < 0.02)
        within_box_y = bool(abs(obj[1] - self.drawer_init_pos[1]) - 0.15 - 0.05 < 0.05)
        success = less_than_z and within_box_x and within_box_y
        # print(less_than_z, within_box_x, within_box_y)
        # in_place = reward_utils.tolerance(
        #     obj_to_target,
        #     bounds=(0, _TARGET_RADIUS),
        #     margin=in_place_margin,
        #     sigmoid="long_tail",
        # )
        return success

        # gripper = obs[:3]
        # handle = obs[4:7]

        # handle_error = np.linalg.norm(handle - self._target_pos)

        # reward_for_opening = reward_utils.tolerance(
        #     handle_error, bounds=(0, 0.02), margin=self.maxDist, sigmoid="long_tail"
        # )

        # handle_pos_init = self._target_pos + np.array([0.0, self.maxDist, 0.0])
        # # Emphasize XY error so that gripper is able to drop down and cage
        # # handle without running into it. By doing this, we are assuming
        # # that the reward in the Z direction is small enough that the agent
        # # will be willing to explore raising a finger above the handle, hook it,
        # # and drop back down to re-gain Z reward
        # scale = np.array([3.0, 3.0, 1.0])
        # gripper_error = (handle - gripper) * scale
        # gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        # reward_for_caging = reward_utils.tolerance(
        #     np.linalg.norm(gripper_error),
        #     bounds=(0, 0.01),
        #     margin=np.linalg.norm(gripper_error_init),
        #     sigmoid="long_tail",
        # )

        # reward = reward_for_caging + reward_for_opening
        # reward *= 5.0

        # # for moving the block
        # tcp = self.tcp_center
        # obj = obs[11:14]
        # tcp_opened = obs[3]
        # target = self._target_pos_block

        # obj_to_target = np.linalg.norm(obj - target)
        # tcp_to_obj = np.linalg.norm(obj - tcp)
        # in_place_margin = np.linalg.norm(self.block_init_pos - target)

        # in_place = reward_utils.tolerance(
        #     obj_to_target,
        #     bounds=(0, _TARGET_RADIUS),
        #     margin=in_place_margin,
        #     sigmoid="long_tail",
        # )

        # object_grasped = self._gripper_caging_reward(
        #     action=action,
        #     obj_pos=obj,
        #     obj_radius=0.02,
        #     pad_success_thresh=0.05,
        #     object_reach_radius=0.01,
        #     xz_thresh=0.01,
        #     high_density=False,
        # )

        # return (
        #     reward,
        #     np.linalg.norm(handle - gripper),
        #     obs[3],
        #     in_place,
        #     reward_for_caging,
        #     reward_for_opening,
        # )
