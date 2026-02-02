# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from .biped300125_asset_cfg import BIPED300125_CFG

##
# Rewards
##


@configclass
class Biped300125RewardsCfg(RewardsCfg):
    """Reward terms for the biped locomotion task."""

    # --- rewards (tracking)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # --- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="Revolute.*")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # --- gait terms
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*lower_1"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*lower_1"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*lower_1"),
        },
    )


##
# Environment configuration
##


@configclass
class Biped300125RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Velocity-tracking locomotion configuration for the custom biped."""

    rewards: Biped300125RewardsCfg = Biped300125RewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = BIPED300125_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/bipad_walker_final_y4/base_link"
        # Contact sensor prims live under an extra /World/<robot_name> scope in this USD.
        # Point the contact sensor directly to the robot link scope so it can find rigid bodies.
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/bipad_walker_final_y4/.*"

        # Actions
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.joint_names = ["Revolute.*"]

        # Randomization (disabled for initial training)
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None

        # Resets
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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

        # Rewards: disable unused standard terms
        self.rewards.undesired_contacts = None
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_pos_limits.weight = 0.0
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="Revolute.*")

        # Commands
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)  # direction_change
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  # direction_change
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)  # direction_change
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Mixed rough terrain (includes random_uniform)
        self.scene.terrain.terrain_type = "generator"  #terrrain _change
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG  #terrrain _change
        self.scene.terrain.max_init_terrain_level = 5  #terrrain _change
        self.scene.height_scanner = self.scene.height_scanner  #terrrain _change
        self.observations.policy.height_scan = self.observations.policy.height_scan  #terrrain _change
        self.curriculum.terrain_levels = self.curriculum.terrain_levels  #terrrain _change


@configclass
class Biped300125RoughEnvCfg_PLAY(Biped300125RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
     # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
