# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

YX_new_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/YX_1/YX_new_usd/YX_new.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.73),
        joint_pos={
            ".*_pitch_joint": -0.1,
            ".*_knee_joint": 0.55,
            ".*_ankle_joint": -0.45,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_yaw_joint",
                ".*_roll_joint",
                ".*_pitch_joint",
                ".*_knee_joint",
            ],

            effort_limit_sim={
                ".*_yaw_joint": 20.0,
                ".*_roll_joint": 33.50,
                ".*_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_yaw_joint": 20.94,
                ".*_roll_joint": 21,
                ".*_pitch_joint": 20.94,
                ".*_knee_joint": 20.94,
            },
            stiffness={
                ".*_yaw_joint": 10.0,
                ".*_roll_joint": 15.0,
                ".*_pitch_joint": 10.0,
                ".*_knee_joint": 10.0,
            },
            damping={
                ".*_yaw_joint": 0.5,
                ".*_roll_joint": 0.5,
                ".*_pitch_joint": 0.5,
                ".*_knee_joint": 1.0,
            },
            armature=0.01,

        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit_sim={".*_ankle_joint": 12.0},
            velocity_limit_sim={".*_ankle_joint": 31.0},
            
            stiffness={".*_ankle_joint": 5.0},
            damping={".*_ankle_joint": 0.5},
            armature=0.01,

        ),
      
    },
)

