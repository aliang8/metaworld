import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from garage.np import discount_cumsum, stack_tensor_dict_list


class SawyerDrawerPutBlockV2Policy(Policy):
    def __init__(self):
        super().__init__()

        self.opened_drawer = False
        self.done_moving_up = False
        # self.moved_block = False
        # self.moving_block = False
        # self.ready_to_drop_block = False

    @staticmethod
    # @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "drwr_pos": obs[4:7],
            "drwr_rot": obs[7:11],
            "block_pos": obs[11:14],
            "block_rot": obs[14:18],
            "unused_info": obs[18:],
            "target_pos": obs[-3:],
        }

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d["hand_pos"]
        pos_cube = o_d["block_pos"] + np.array([-0.005, 0.0, 0.015])
        gripper_separation = o_d["gripper"]
        pos_bin = o_d["drwr_pos"] + np.array([0.0, 0.1, 0.0])

        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04:
            # print("moving above the block")
            return pos_cube + np.array([0.0, 0.0, 0.3])
        elif abs(pos_curr[2] - pos_cube[2]) > 0.04:
            # print("going to pick up the block")
            return pos_cube
        elif gripper_separation > 0.73:
            return pos_curr
        elif np.linalg.norm(pos_curr[:2] - pos_bin[:2]) > 0.02:
            # print("moving block to drawer")
            if pos_curr[2] < 0.3:
                return pos_curr + np.array([0.0, 0.0, 0.3])
            return np.array([*pos_bin[:2], 0.18])
        else:
            return pos_bin

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d["hand_pos"]
        pos_cube = o_d["block_pos"]

        if (
            np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04
            or abs(pos_curr[2] - pos_cube[2]) > 0.15
        ):
            return -1.0
        else:
            return 0.6

    def reset(self):
        self.opened_drawer = False
        self.done_moving_up = False

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        # set original object poses
        if not hasattr(self, "orig_drawer_pos"):
            self.orig_drawer_pos = o_d["drwr_pos"]
            self.orig_block_pos = o_d["block_pos"]

        # print(
        #     "\thand_pose: ",
        #     o_d["hand_pos"],
        #     " \n\tdrawer_pose: ",
        #     o_d["drwr_pos"],
        #     " \n\tblock_pose: ",
        #     o_d["block_pos"],
        #     " \n\tgripper: ",
        #     o_d["gripper"],
        # )

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # NOTE this policy looks different from the others because it must
        # modify its p constant part-way through the task
        pos_curr = o_d["hand_pos"]
        pos_drwr = o_d["drwr_pos"] + np.array([0.0, 0.0, -0.02])

        if not self.opened_drawer:
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
                # print("aligning end effector z")
                to_pos = pos_drwr + np.array([0.0, 0.0, 0.3])
                action["delta_pos"] = move(o_d["hand_pos"], to_pos, p=4.0)

            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                # print("dropping down to touch handle")
                to_pos = pos_drwr
                action["delta_pos"] = move(o_d["hand_pos"], to_pos, p=4.0)

            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                # print("pulling handle")
                to_pos = pos_drwr + np.array([0.0, -0.08, 0.0])
                action["delta_pos"] = move(o_d["hand_pos"], to_pos, p=10.0)

                # if it pulled drawer beyond certain point, stop
                if abs(pos_drwr[1] - self.orig_drawer_pos[1]) > 0.08:
                    print("done pulling handle")
                    self.opened_drawer = True

        if self.opened_drawer:
            # move arm up
            if abs(pos_curr[2] - 0.3) > 0.04 and not self.done_moving_up:
                # print("moving up")
                to_pos = pos_curr + np.array([0.0, 0.0, 0.3])
                action["delta_pos"] = move(o_d["hand_pos"], to_pos, p=4.0)
            else:
                self.done_moving_up = True

            if self.done_moving_up:
                # move towards block
                action["delta_pos"] = move(
                    o_d["hand_pos"], self._desired_pos(o_d), p=25.0
                )
                action["grab_effort"] = self._grab_effort(o_d)

                # check done condition and then open gripper
        else:
            # keep gripper open
            action["grab_effort"] = -1.0

        return action.array
