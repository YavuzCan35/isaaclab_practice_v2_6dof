# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import mdp as env_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from .biped_v2_6dof_asset_cfg import BIPED_V2_6DOF_CFG

##
# Rewards
##


@configclass
class BipedV26dofRewardsCfg(RewardsCfg):
    """Reward terms for the v2 6-DoF biped locomotion task."""

    # --- rewards (tracking)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.9,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.3,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # --- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="Revolute.*")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # --- gait terms
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight= 0.6,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*lower_.*"),
            "threshold": 0.2,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight= -0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*lower_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*lower_.*"),
        },
    )


##
# Environment configuration
##


@configclass
class BipedV26dofEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Velocity-tracking locomotion configuration for the v2 6-DoF biped."""

    rewards: BipedV26dofRewardsCfg = BipedV26dofRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = BIPED_V2_6DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/.*/base_link"
        if self.scene.contact_forces is not None:
            # Limit to links that actually have PhysxContactReportAPI.
            self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*/(base_link|lower_.*)"

        # Actions
        self.actions.joint_pos.joint_names = [
            "Revolute_hip_left",
            "Revolute_upper_left_leg",
            "Revolute_lower_left_leg",
            "Revolute_hip_right",
            "Revolute_upper_right_leg",
            "Revolute_lower_right_leg",
        ]
        self.actions.joint_pos.scale = {
            "Revolute_hip_left": 0.5,
            "Revolute_upper_left_leg": -0.5,
            "Revolute_lower_left_leg": -0.5,
            "Revolute_hip_right": 0.5,
            "Revolute_upper_right_leg": 0.5,
            "Revolute_lower_right_leg": 0.5,
        }

        # Randomization (disabled for initial training)
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None

        # Resets
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]
        self.terminations.bad_orientation = DoneTerm(
            func=env_mdp.bad_orientation,
            params={"limit_angle": 0.6, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Rewards: disable unused standard terms
        self.rewards.undesired_contacts = None
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.dof_pos_limits.weight = 0.0
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="Revolute.*")

        # Commands: slight flexibility to encourage stepping
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None

        # Flat ground only
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class BipedV26dofEnvCfg_PLAY(BipedV26dofEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 2.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
