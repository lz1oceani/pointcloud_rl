import pathlib

import numpy as np
import trimesh
from mani_skill.env.base_env import BaseEnv
from mani_skill.utils.contrib import angle_distance, apply_pose_to_points, norm, normalize_and_clip_in_interval, o3d_to_trimesh, trimesh_to_o3d
from mani_skill.utils.geometry import get_axis_aligned_bbox_for_articulation, rotate_2d_vec_by_angle
from mani_skill.utils.misc import sample_from_tuple_or_scalar
from mani_skill.utils.o3d_utils import merge_mesh, np2mesh
from sapien.core import Articulation, Pose

_this_file = pathlib.Path(__file__).resolve()


class OpenCabinetEnvBase(BaseEnv):
    def __init__(
        self,
        yaml_file_path,
        fixed_target_link_id=None,
        joint_friction=(0.05, 0.15),
        joint_damping=(5, 20),
        l1_rew=False,
        require_grasp=False,
        joint_stiffness=None,
        reward_mode="new",
        *args,
        **kwargs,
    ):

        self.joint_friction = joint_friction
        self.joint_damping = joint_damping
        self.joint_stiffness = joint_stiffness
        self.once_init = False
        self.last_id = None
        self.l1_rew = l1_rew
        self.reward_mode = reward_mode
        self.require_grasp = require_grasp
        kwargs = dict(kwargs)
        kwargs["vhacd_mode"] = kwargs.get("vhacd_mode", "cvx")
        self.fixed_target_link_id = fixed_target_link_id
        super().__init__(
            _this_file.parent.joinpath(yaml_file_path),
            *args,
            **kwargs,
        )

    def configure_env(self):
        self.cabinet_max_dof = 8  # actually, it is 6 for our data

    def get_visual_state(self):
        joint_pos = self.cabinet.get_qpos()[self.target_index_in_active_joints] / self.target_qpos
        current_handle = apply_pose_to_points(self.o3d_info[self.target_link_name][-1], self.target_link.get_pose()).mean(0)

        if self.be_ego_mode:
            mat = self.get_to_ego_pose().to_transformation_matrix()
            current_handle = np.concatenate([current_handle, np.ones(1)], axis=-1)
            current_handle = (mat @ current_handle)[:3]

        flag = self.compute_other_flag_dict()
        # dist_ee_to_handle = flag['dist_ee_to_handle']
        # dist_ee_mid_to_handle = flag['dist_ee_mid_to_handle']
        ee_close_to_handle = flag["ee_close_to_handle"]
        # print(dist_ee_mid_to_handle, dist_ee_mid_to_handle, ee_close_to_handle)
        # dist_ee_to_handle, dist_ee_mid_to_handle,

        # , float(ee_close_to_handle)
        return np.concatenate([np.array([joint_pos, joint_pos > 1.0, ee_close_to_handle]), current_handle], axis=0).astype(np.float32)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.cabinet: Articulation = self.articulations["cabinet"]["articulation"]

        self._place_cabinet()
        self._close_all_parts()
        self._find_handles_from_articulation()
        self._place_robot()
        self._choose_target_link()
        self._ignore_collision()
        self._set_joint_physical_parameters()
        self._prepare_for_obs()

        [[lmin, lmax]] = self.target_joint.get_limits()

        self.target_qpos = lmin + (lmax - lmin) * self.custom["open_extent"]
        self.pose_at_joint_zero = self.target_link.get_pose()
        self.init_arm_qpos = self.agent.get_state(by_dict=True, with_controller_state=False)["qpos"][1:-3]
        # print(self.agent.get_state(by_dict=True, with_controller_state=False)["qpos"])
        # print(len(self.init_arm_qpos))
        # exit(0)
        return self.get_obs()

    def _place_cabinet(self):
        mins, maxs = get_axis_aligned_bbox_for_articulation(self.cabinet)
        self.cabinet.set_pose(Pose(np.array([0, 0, -mins[2]]) + self.delta_pos, [1, 0, 0, 0]))

    def _find_handles_from_articulation(self):
        handles_info = {}
        handles_visual_body_ids = {}
        o3d_info = {}
        grasp_pose = {}

        for link in self.cabinet.get_links():
            link_name = link.get_name()
            assert link_name not in handles_info
            handles_info[link_name] = []
            handles_visual_body_ids[link_name] = []

            o3d_info[link_name] = []
            for visual_body in link.get_visual_bodies():
                if "handle" not in visual_body.get_name():
                    continue
                handles_visual_body_ids[link_name].append(visual_body.get_visual_id())
                for i in visual_body.get_render_shapes():
                    vertices = apply_pose_to_points(i.mesh.vertices * visual_body.scale, visual_body.local_pose)
                    mesh = np2mesh(vertices, i.mesh.indices.reshape(-1, 3))
                    o3d_info[link_name].append(mesh)
                    handles_info[link_name].append((i.mesh.vertices * visual_body.scale, i.mesh.indices, visual_body.local_pose))
            if len(handles_info[link_name]) == 0:
                handles_info.pop(link_name)
                handles_visual_body_ids.pop(link_name)
                o3d_info.pop(link_name)

        for link in self.cabinet.get_links():
            link_name = link.get_name()
            if link_name not in o3d_info:
                continue
            mesh = merge_mesh(o3d_info[link_name])

            mesh = trimesh.convex.convex_hull(o3d_to_trimesh(mesh))
            pcd = mesh.sample(500)
            pcd_world = apply_pose_to_points(pcd, link.get_pose())
            lens = (pcd_world.max(0) - pcd_world.min(0)) / 2
            center = (pcd_world.max(0) + pcd_world.min(0)) / 2
            box_size = lens / 2
            # print(box_size, center)
            # exit(0)

            if lens[1] > lens[2]:
                flat = np.array([0, 0, 1])
            else:
                flat = np.array([0, 1, 0])
            # print(center, box_size, lens)

            def crop_by_box():
                region0, region1 = center.copy(), center.copy()
                region1[0] += lens[0]
                region0[0] -= lens[0]
                if lens[1] > lens[2]:
                    region0[1] -= box_size[1]
                    region0[2] -= lens[2]

                    region1[1] += box_size[1]
                    region1[2] += lens[2]
                else:
                    region0[1] -= lens[1]
                    region0[2] -= box_size[2]

                    region1[1] += lens[1]
                    region1[2] += box_size[2]

                sign = np.logical_and(region0 <= pcd_world, pcd_world <= region1)
                sign = np.all(sign, axis=-1)
                return pcd_world[sign]

            pcd_world = crop_by_box()

            if pcd_world.shape[0] > 100:
                pcd_world = pcd_world[:100]
            pcd = apply_pose_to_points(pcd_world, link.get_pose().inv())

            def build_pose(forward, flat):
                extra = np.cross(flat, forward)
                ans = np.eye(4)
                ans[:3, :3] = np.array([extra, flat, forward]).T
                return Pose.from_transformation_matrix(ans)

            grasp_pose[link_name] = (link.get_pose().inv() * build_pose([1, 0, 0], flat), link.get_pose().inv() * build_pose([1, 0, 0], -flat))
            o3d_info[link_name] = (link, trimesh_to_o3d(mesh), pcd)

        self.handles_info = handles_info
        self.handles_visual_body_ids = handles_visual_body_ids
        self.o3d_info = o3d_info
        self.grasp_pose = grasp_pose
        assert len(self.handles_info.keys()) > 0

    def _close_all_parts(self):
        qpos = []
        for joint in self.cabinet.get_active_joints():
            [[lmin, lmax]] = joint.get_limits()
            if lmin == -np.inf or lmax == np.inf:
                raise Exception("This object has an inf limit joint.")
            qpos.append(lmin)
        self.cabinet.set_qpos(np.array(qpos))

    def _choose_target_link(self, joint_type):
        links, joints = [], []
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if joint.type == joint_type and link.get_name() in self.handles_info:
                links.append(link)
                joints.append(joint)

        if self.fixed_target_link_id is not None:
            self.target_index = self.fixed_target_link_id % len(joints)
        else:
            self.target_index = self._level_rng.choice(len(joints))  # only sample revolute/prismatic joints
            # the above line will change leve_rng's internal state multiple times
        self.target_link = links[self.target_index]
        self.target_link_name = self.target_link.get_name()
        self.target_joint = joints[self.target_index]
        self.target_index_in_active_joints = self.cabinet.get_active_joints().index(self.target_joint)
        self.target_indicator = np.zeros(self.cabinet_max_dof)
        self.target_indicator[self.target_index_in_active_joints] = 1

    def _place_robot(self):
        # negative x is door/drawer
        # base pos
        center = np.array([0, 0.8])  # to make the gripper closer to the cabinet
        dist = self._level_rng.uniform(low=1.6, high=1.8)
        # dist = self._level_rng.uniform(low=1, high=1.1)

        theta = self._level_rng.uniform(low=0.9 * np.pi, high=1.1 * np.pi)
        delta = np.array([np.cos(theta), np.sin(theta)]) * dist
        base_pos = center + delta + self.delta_pos[:2]

        # base orientation
        perturb_orientation = self._level_rng.uniform(low=-0.05 * np.pi, high=0.05 * np.pi)
        base_theta = -np.pi + theta + perturb_orientation

        self.agent.set_state(
            {
                "base_pos": base_pos,
                "base_orientation": base_theta,
            },
            by_dict=True,
        )

    def _ignore_collision(self):
        """ignore collision among all movable links"""
        cabinet = self.cabinet
        for joint, link in zip(cabinet.get_joints(), cabinet.get_links()):
            if joint.type in ["revolute", "prismatic"]:
                shapes = link.get_collision_shapes()
                for s in shapes:
                    g0, g1, g2, g3 = s.get_collision_groups()
                    s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)

    def _prepare_for_obs(self):
        self.handle_visual_ids = self.handles_visual_body_ids[self.target_link.get_name()]
        self.target_link_ids = [self.target_link.get_id()]

    def get_additional_task_info(self, obs_mode):
        if obs_mode == "state":
            return self.target_indicator
        else:
            return np.array([])

    def get_all_objects_in_state(self):
        return [], [(self.cabinet, self.cabinet_max_dof)]

    def _set_joint_physical_parameters(self):
        for joint in self.cabinet.get_joints():
            if self.joint_friction is not None:
                joint.set_friction(sample_from_tuple_or_scalar(self._level_rng, self.joint_friction))
            if self.joint_damping is not None:
                joint.set_drive_property(
                    stiffness=0, damping=sample_from_tuple_or_scalar(self._level_rng, self.joint_damping), force_limit=3.4028234663852886e38
                )

    def compute_other_flag_dict(self, show=False):
        ee_cords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
        current_handle = apply_pose_to_points(self.o3d_info[self.target_link_name][-1], self.target_link.get_pose())  # [200, 3]
        ee_to_handle = ee_cords[..., None, :] - current_handle
        dist_ee_to_handle = np.linalg.norm(ee_to_handle, axis=-1).min(-1).min(-1)  # [2]

        handle_mesh = trimesh.Trimesh(
            vertices=apply_pose_to_points(np.asarray(self.o3d_info[self.target_link_name][-2].vertices), self.target_link.get_pose()),
            faces=np.asarray(np.asarray(self.o3d_info[self.target_link_name][-2].triangles)),
        )

        dist_ee_mid_to_handle = trimesh.proximity.ProximityQuery(handle_mesh).signed_distance(ee_cords.mean(0)).max()

        ee_close_to_handle = dist_ee_to_handle.max() <= 0.025 and dist_ee_mid_to_handle > 0
        other_info = {
            "dist_ee_to_handle": dist_ee_to_handle,
            "dist_ee_mid_to_handle": dist_ee_mid_to_handle,
            "ee_close_to_handle": ee_close_to_handle,
        }
        other_info["dist_ee_to_handle_l1"] = np.abs(ee_to_handle).sum(axis=-1).min(-1).min(-1)  # [2]

        if show:
            from pyrl.utils.lib3d import np2pcd, to_o3d
            from pyrl.utils.visualization import visualize_3d

            print(dist_ee_to_handle, dist_ee_mid_to_handle)
            visualize_3d([to_o3d(handle_mesh), np2pcd(ee_cords.reshape(-1, 3))])

        return other_info

    def compute_eval_flag_dict(self):
        flag_dict = {
            "cabinet_static": self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1),
            "open_enough": self.cabinet.get_qpos()[self.target_index_in_active_joints] >= self.target_qpos,
        }
        if self.require_grasp:
            flag_dict["gasp_handle"] = self.compute_other_flag_dict()["ee_close_to_handle"]
        flag_dict["success"] = all(flag_dict.values())
        return flag_dict

    def old_reward(self, action, state):
        if state is not None:
            self.set_state(state)

        actor = self.target_link

        flag_dict = self.compute_eval_flag_dict()
        other_info = self.compute_other_flag_dict()
        dist_ee_to_handle = other_info["dist_ee_to_handle_l1"] if self.l1_rew else other_info["dist_ee_to_handle"]
        dist_ee_mid_to_handle = other_info["dist_ee_mid_to_handle"]

        agent_pose = self.agent.hand.get_pose()
        target_pose = self.target_link.get_pose() * self.grasp_pose[self.target_link_name][0]
        target_pose_2 = self.target_link.get_pose() * self.grasp_pose[self.target_link_name][1]

        angle1 = angle_distance(agent_pose, target_pose)
        angle2 = angle_distance(agent_pose, target_pose_2)
        gripper_angle_err = min(angle1, angle2) / np.pi

        cabinet_vel = self.cabinet.get_qvel()[self.target_index_in_active_joints]

        gripper_vel_norm = min(norm(actor.get_velocity()), 1)
        gripper_ang_vel_norm = min(norm(actor.get_angular_velocity()), 1)

        scale = 1
        reward = 0
        stage_reward = 0

        vel_coefficient = 1.5
        dist_coefficient = 0.5

        gripper_angle_rew = -gripper_angle_err * 3

        rew_ee_handle = -dist_ee_to_handle.mean() * 2
        rew_ee_mid_handle = normalize_and_clip_in_interval(dist_ee_mid_to_handle, -0.01, 4e-3) - 1

        reward = gripper_angle_rew + rew_ee_handle + rew_ee_mid_handle - (dist_coefficient + vel_coefficient)
        stage_reward = -(5 + vel_coefficient + dist_coefficient)

        vel_reward = 0
        dist_reward = 0

        if other_info["ee_close_to_handle"]:
            stage_reward += 0.5
            vel_reward = normalize_and_clip_in_interval(cabinet_vel, -0.1, 0.5) * vel_coefficient  # Push vel to positive
            dist_reward = (
                normalize_and_clip_in_interval(self.cabinet.get_qpos()[self.target_index_in_active_joints], 0, self.target_qpos) * dist_coefficient
            )
            reward += dist_reward + vel_reward
            if flag_dict["open_enough"]:
                stage_reward += vel_coefficient + 2
                gripper_vel_rew = -(gripper_vel_norm + gripper_ang_vel_norm * 0.5)
                reward = reward - vel_reward + gripper_vel_rew
                if flag_dict["cabinet_static"]:
                    stage_reward += 1
        info_dict = {
            "dist_ee_to_handle": dist_ee_to_handle.mean(),
            "angle1": angle1,
            "angle2": angle2,
            "dist_ee_mid_to_handle": dist_ee_mid_to_handle,
            "rew_ee_handle": rew_ee_handle,
            "rew_ee_mid_handle": rew_ee_mid_handle,
            "qpos_rew": dist_reward,
            "qvel_rew": vel_reward,
            "gripper_angle_err": gripper_angle_err * 180,
            "gripper_angle_rew": gripper_angle_rew,
            "gripper_vel_norm": gripper_vel_norm,
            "gripper_ang_vel_norm": gripper_ang_vel_norm,
            "qpos": self.cabinet.get_qpos()[self.target_index_in_active_joints],
            "qvel": cabinet_vel,
            "target_qpos": self.target_qpos,
            "reward_raw": reward,
            "stage_reward": stage_reward,
            "ee_close_to_handle": float(other_info["ee_close_to_handle"]),
            "open_enough": float(flag_dict["open_enough"]),
        }
        reward = (reward + stage_reward) * scale
        return reward, info_dict

    def compute_other_flag_dict_new(self, show=False):
        ee_cords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
        current_handle = apply_pose_to_points(self.o3d_info[self.target_link_name][-1], self.target_link.get_pose())  # [200, 3]
        ee_to_handle = ee_cords.mean(0)[:, None] - current_handle
        dist_ee_to_handle = np.linalg.norm(ee_to_handle, axis=-1).min(-1).mean(-1)

        handle_mesh = trimesh.Trimesh(
            vertices=apply_pose_to_points(np.asarray(self.o3d_info[self.target_link_name][-2].vertices), self.target_link.get_pose()),
            faces=np.asarray(np.asarray(self.o3d_info[self.target_link_name][-2].triangles)),
        )

        # 0.04 / 10 = 4mm
        # num = 10
        # points_between_ee = np.concatenate([ee_cords[0] * i / num + ee_cords[1] * (num - i) / num for i in range(num)], axis=0)
        handle_obj = trimesh.proximity.ProximityQuery(handle_mesh)
        sd_ee_mid_to_handle = handle_obj.signed_distance(ee_cords.mean(0)).max()
        sd_ee_to_handle = handle_obj.signed_distance(ee_cords.reshape(-1, 3)).reshape(2, -1).max(1)
        # abs_sd_ee_to_handle = np.abs(abs_sd_ee_to_handle)

        # Grasp = mid close almost in cvx and both sides has similar sign distance.
        close_to_grasp = sd_ee_to_handle.min() > -1e-2
        ee_in_grasp_pose = sd_ee_mid_to_handle > -1e-2
        grasp_happen = ee_in_grasp_pose and close_to_grasp

        other_info = {
            "dist_ee_to_handle": dist_ee_to_handle,
            "sd_ee_mid_to_handle": sd_ee_mid_to_handle,
            "sd_ee_to_handle": sd_ee_to_handle,
            "ee_close_to_handle_pre_rot": dist_ee_to_handle <= 0.06,
            "ee_close_to_handle": dist_ee_to_handle <= 0.03,
            "grasp_happen": grasp_happen,
        }
        other_info["dist_ee_to_handle_l1"] = np.abs(ee_to_handle).sum(axis=-1).min(-1).min(-1)  # [2]

        if show:
            from pyrl.utils.lib3d import np2pcd, to_o3d
            from pyrl.utils.visualization import visualize_3d

            print(dist_ee_to_handle, sd_ee_mid_to_handle, sd_ee_to_handle)
            visualize_3d([to_o3d(handle_mesh), np2pcd(ee_cords.reshape(-1, 3))])

        return other_info

    def new_reward(self, action, state):
        if state is not None:
            self.set_state(state)

        actor = self.target_link
        gripper_action = action[-2:]
        robot_qpos = self.agent.get_state(by_dict=True, with_controller_state=False)["qpos"]
        gripper_qpos = robot_qpos[-2:]
        # def get_state(self, by_dict=False, with_controller_state=True):

        flag_dict = self.compute_eval_flag_dict()
        other_info = self.compute_other_flag_dict_new()
        dist_ee_to_handle = other_info["dist_ee_to_handle"]
        sd_ee_mid_to_handle = other_info["sd_ee_mid_to_handle"]  # > 0 means good

        cabinet_qpos = self.cabinet.get_qpos()[self.target_index_in_active_joints]
        cabinet_qvel = self.cabinet.get_qvel()[self.target_index_in_active_joints]

        agent_pose = self.agent.hand.get_pose()
        agent_vel = self.agent.hand.get_velocity()
        target_pose = self.target_link.get_pose() * self.grasp_pose[self.target_link_name][0]
        target_pose_2 = self.target_link.get_pose() * self.grasp_pose[self.target_link_name][1]
        target_vel = (target_pose.to_transformation_matrix()[:3, :3] @ np.array([0, 0, -1]))[:2]

        if not (self.be_ego_mode is False):
            target_action = rotate_2d_vec_by_angle(target_vel, -self.agent._get_base_orientation())
        else:
            target_action = target_vel
        base_action_err = -np.linalg.norm(action[:2] - target_action)
        gripper_vel_rew = -np.linalg.norm(agent_vel[:2] - target_vel)

        # print(vel_mag, agent_vel, target_vel)
        # exit(0)

        angle1 = np.abs(angle_distance(agent_pose, target_pose))
        angle2 = np.abs(angle_distance(agent_pose, target_pose_2))
        gripper_angle_err = min(angle1, angle2) / np.pi
        # if gripper_angle_err > 1:
        #     print(
        #         "Error", angle1, angle2, angle_distance(agent_pose, target_pose), angle_distance(agent_pose, target_pose_2), agent_pose, target_pose_2
        #     )
        #     exit(0)

        # gripper [0, 0.04] close -> open
        open_gripper_rew = 10 * gripper_qpos.mean()  # Keep open [0, 0.4]
        close_gripper_rew = (
            -10 * gripper_qpos.mean() + 0.45  # - (np.clip(gripper_action, a_min=-1, a_max=1).mean() - 1) * 0.1
        )  # Keep open [0, 0.2] + [0, 0.5]

        open_cabinet_reward = 0
        static_reward = 0
        gipeer_vel_rew = 0
        keep_static_reward = 0

        # Before grasp
        arm_pos_err = np.abs(self.init_arm_qpos - robot_qpos[1:-3]).mean()
        keep_arm_rew = -arm_pos_err - np.abs(action[4:-2]).mean()
        close_to_cabinet_rew = (
            -np.clip(gripper_angle_err, a_min=1 / 12.0, a_max=1) * 1.5 - dist_ee_to_handle.mean() * 2 + sd_ee_mid_to_handle
        )  # move to grasp. [-oo, 0]
        good_pose_rew = -np.clip(gripper_angle_err, a_min=1 / 12.0, a_max=1) * 0.4 + 0.4  # [0.1, 0.6]

        gipper_rew = open_gripper_rew
        stage_index = 0
        grasp_reward = 0
        # if other_info["ee_close_to_handle_pre_rot"]:
        #     stage_index = 1
        #     gipper_rew = open_gripper_rew + good_pose_rew

        if gripper_angle_err * 180 <= 25 and other_info["ee_close_to_handle"]:
            stage_index = 2
            gipper_rew = close_gripper_rew + good_pose_rew
            if other_info["grasp_happen"]:
                # grasp_reward = 0.5
                stage_index = 3
                gipeer_vel_rew = np.clip(base_action_err + gripper_vel_rew, a_min=-2, a_max=0) + 2  # [0, 2]
                close_to_cabinet_rew = 0.1
                # print(base_action_err, base_action, target_vel, self.agent._get_base_orientation(), self.be_ego_mode)
                # exit(0)
                keep_arm_rew = 0
                open_cabinet_reward = (
                    normalize_and_clip_in_interval(cabinet_qpos, 0, self.target_qpos * 1.1) + np.clip(cabinet_qvel, a_min=-0.5, a_max=0.5) + 0.5
                )  # [0, 2]
                if flag_dict["open_enough"]:
                    stage_index = 4
                    gipeer_vel_rew = 2.5
                    keep_arm_rew = 0
                    open_cabinet_reward = 2  # [0, 2]
                    static_reward = (-np.clip(np.abs(action), a_min=0, a_max=1).mean() + 1) * 2  # [0, 1]
                    if flag_dict["cabinet_static"]:
                        stage_index = 5
                        keep_static_reward += 1

        reward = (
            close_to_cabinet_rew
            + keep_arm_rew
            + gipper_rew
            + gipeer_vel_rew
            + open_cabinet_reward
            + static_reward
            + keep_static_reward
            + grasp_reward
        )

        info_dict = {
            "angle1": angle1,
            "angle2": angle2,
            "gripper_action": action[-2:],
            "agent_vel": agent_vel,
            "target_action": target_action,
            "base_action": action[:2],
            "base_action_err": base_action_err,
            "open_gripper_rew": open_gripper_rew,
            "close_gripper_rew": close_gripper_rew,
            "good_pose_rew": good_pose_rew,
            "gripper_qpos": gripper_qpos,
            "dist_ee_to_handle": dist_ee_to_handle,
            "sd_ee_mid_to_handle": sd_ee_mid_to_handle,
            "sd_ee_to_handle": other_info["sd_ee_to_handle"].min(),
            "gripper_angle_err": gripper_angle_err * 180,
            "to_cabinet_rew": close_to_cabinet_rew,
            "gipper_rew": gipper_rew,
            "keep_arm_rew": keep_arm_rew,
            "gipeer_vel_rew": gipeer_vel_rew,
            "open_cabinet_reward": open_cabinet_reward,
            "static_reward": static_reward,
            "keep_static_reward": keep_static_reward,
            "qpos": self.cabinet.get_qpos()[self.target_index_in_active_joints],
            "qvel": self.cabinet.get_qvel()[self.target_index_in_active_joints],
            "target_qpos": self.target_qpos,
            "reward_raw": reward,
            "ee_close_to_handle_pre_rot": float(other_info["ee_close_to_handle_pre_rot"]),
            "ee_close_to_handle": float(other_info["ee_close_to_handle"]),
            "grasp_happen": float(other_info["grasp_happen"]),
            "open_enough": float(flag_dict["open_enough"]),
            "cabinet_static": float(flag_dict["cabinet_static"]),
            "single_success": float(flag_dict["success"]),
            "stage_index": stage_index,
        }
        return reward, info_dict

    def compute_dense_reward(self, action, state=None):
        if self.reward_mode == "old":
            return self.old_reward(action, state)
        else:
            return self.new_reward(action, state)

    def _post_process_view(self, view_dict):
        visual_id_seg = view_dict["seg"][..., 0]  # (n, m)
        actor_id_seg = view_dict["seg"][..., 1]  # (n, m)

        masks = [np.zeros(visual_id_seg.shape, dtype=np.bool_) for _ in range(3)]

        for visual_id in self.handle_visual_ids:
            masks[0] = masks[0] | (visual_id_seg == visual_id)
        for actor_id in self.target_link_ids:
            masks[1] = masks[1] | (actor_id_seg == actor_id)
        for actor_id in self.robot_link_ids:
            masks[2] = masks[2] | (actor_id_seg == actor_id)

        view_dict["seg"] = np.stack(masks, axis=-1)

    def num_target_links(self, joint_type):
        links, joints = [], []
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if joint.type == joint_type and link.get_name() in self.handles_info:
                links.append(link)
                joints.append(joint)
        return len(links)

    def get_inst_labels(self, part_masks, link_masks):
        if self.selected_id not in self.bbox_cache:
            self.bbox_cache[self.selected_id] = {}
            for link in self.cabinet.get_links():
                if link.get_name() not in self.handles_info:
                    continue
                handle_name = []
                board_name = []
                trash_name = []
                handle_id = []
                board_id = []

                for visual_body in link.get_visual_bodies():
                    part_name = visual_body.get_name()
                    if "handle" in part_name:
                        handle_id.append(visual_body.get_visual_id())
                        handle_name.append(part_name)
                    elif "drawer_front" in part_name or "door_surface" in part_name:
                        board_id.append(visual_body.get_visual_id())
                        board_name.append(part_name)
                    else:
                        trash_name.append(part_name)

                from pyrl.utils.sapien import actor_to_bbox

                handle_bbox = actor_to_bbox(link, part_exclude=trash_name + board_name, box_type="obb", to_vector=True)[:2]
                board_bbox = actor_to_bbox(link, part_exclude=trash_name + handle_name, box_type="obb", to_vector=True)[:2]

                center = np.stack([board_bbox[0], handle_bbox[0]], axis=0)
                size = np.stack([board_bbox[1], handle_bbox[1]], axis=0)

                inv_pose = link.get_pose().inv().to_transformation_matrix()

                center = center @ inv_pose[:3, :3].T + inv_pose[:3, 3]
                classes = np.array([0, 1], dtype=np.uint8)[..., None]

                self.bbox_cache[self.selected_id][link.get_name()] = [center, size, classes, np.array(board_id), np.array(handle_id)]

                # from pyrl.utils.lib3d import np2pcd, create_obb
                # from pyrl.utils.visualization import visualize_3d
                # from pyrl.utils.visualization.color import NYU40_COLOR_PALETTE
                # print(center, size, classes, NYU40_COLOR_PALETTE[classes[0] + 1], NYU40_COLOR_PALETTE[classes[1] + 1])
                # NYU40_COLOR_PALETTE = np.array(NYU40_COLOR_PALETTE)
                # bbox = []
                # for i in range(len(center)):
                #     bbox.append(create_obb(np.concatenate([center[i], size[i]]), link.get_pose().to_transformation_matrix()[:3, :3], NYU40_COLOR_PALETTE[classes[i] + 1] / 255.0))
                # visualize_3d(bbox)

        # link_masks = []
        # masks = [link_masks, ]

        to_ego_mode = self.get_to_ego_pose() if self.be_ego_mode else Pose()

        mask_idx = 0
        masks = np.ones(link_masks.shape, dtype=np.uint8) * 255
        bbox = []

        for link in self.cabinet.get_links():
            if link.get_name() not in self.handles_info:
                continue
            # print(self.selected_id, link.get_name())
            # print(self.selected_id, link.get_name())
            center, size, classes, board_id, handle_id = self.bbox_cache[self.selected_id][link.get_name()]
            pose = (to_ego_mode * link.get_pose()).to_transformation_matrix()
            center = center @ pose[:3, :3].T + pose[:3, 3]
            bbox.append([center, size, pose[None, :3, :3].repeat(2, 0), classes])

            masks[(part_masks[..., None] == board_id).any(-1)] = 0
            mask_idx += 1

            masks[(part_masks[..., None] == handle_id).any(-1)] = 1
            mask_idx += 1
        # print("xxx", (masks != 255).sum())

        padded_num = 32  # 5 * 2
        assert mask_idx <= padded_num, f"Before padding, you have {padded_num} bboxes!"

        bbox.append(
            [
                np.zeros((padded_num - mask_idx, 3), dtype=np.float32),
                np.zeros((padded_num - mask_idx, 3), dtype=np.float32),
                np.zeros((padded_num - mask_idx, 3, 3), dtype=np.float32),
                np.ones((padded_num - mask_idx, 1), dtype=np.uint8) * 255,
            ]
        )

        bbox = [np.concatenate([_[i] for _ in bbox], axis=0) for i in range(4)]
        return bbox, masks

    def get_target_inst_labels(self, link_masks, part_masks):
        assert self.selected_id in self.bbox_cache
        bbox = self.bbox_cache[self.selected_id][self.target_link_name]
        center, size, classes, other_id, handle_id = bbox
        to_ego_mode = self.get_to_ego_pose() if self.be_ego_mode else Pose()
        pose = (to_ego_mode * self.target_link.get_pose()).to_transformation_matrix()
        center = center @ pose[:3, :3].T + pose[:3, 3]
        bbox = (center, size, pose[None, :3, :3].repeat(2, 0), classes)

        # from pyrl.utils.data import GDict
        # bbox = [np.concatenate([_[i] for _ in bbox], axis=0) for i in range(4)]

        masks = np.ones(link_masks.shape, dtype=np.uint8) * 255
        masks[(part_masks[..., None] == other_id).any(-1)] = 0
        masks[(part_masks[..., None] == handle_id).any(-1)] = 1

        return bbox, masks
        # board = cached_bboxes[f'{self.target_link_name.get_name()}-board']
        # handle = cached_bboxes[f'{self.target_link_name.get_name()}-handle']

    # def get_custom_observation(self):
    #     box_corners =
    #     agent_state = self.get_agent_obs()

    #     return np.concatenate()


class OpenCabinetDoorEnv(OpenCabinetEnvBase):
    def __init__(self, *args, split="train", **kwargs):
        super().__init__(f"../assets/config_files/open_cabinet_door.yml", *args, **kwargs)

    def _choose_target_link(self):
        super()._choose_target_link("revolute")

    @property
    def num_target_links(self):
        return super().num_target_links("revolute")


class OpenCabinetDrawerEnv(OpenCabinetEnvBase):
    def __init__(self, yaml_file_path=f"../assets/config_files/open_cabinet_drawer.yml", *args, **kwargs):
        super().__init__(yaml_file_path=yaml_file_path, *args, **kwargs)

    def _choose_target_link(self):
        super()._choose_target_link("prismatic")

    @property
    def num_target_links(self):
        return super().num_target_links("prismatic")
