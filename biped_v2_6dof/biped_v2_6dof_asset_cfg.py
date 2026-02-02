#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset configuration for the v2 6-DoF biped USD."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# NOTE: This task uses the USD directly. The URDF is not referenced in code to avoid mismatches.
BIPED_V2_6DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # USD with PhysxContactReportAPI applied to rigid bodies (required by ContactSensor)
        usd_path=r"C:\Users\firstuser\Downloads\biped_v2_6dof\Biped_v2_6dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Adjust if the robot spawns too high/low.
        pos=(0.0, 0.0, 0.184),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Revolute.*"],
            effort_limit_sim=180.0,
            stiffness=180.0,
            damping=4.0,
        ),
    },
)
