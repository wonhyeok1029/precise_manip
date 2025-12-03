# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math, os, torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg #, DepthCameraCfg, ImageCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, math as math_utils

from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Project root - based on highest directory "precise manip".
# cfg file is in "precise_manip/source/precise_manip/precise_manip/tasks/manager_based/precise_manip/precise_manip_env_cfg.py"
##
current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_file_dir, "../../../../../.."))
print(f"[INFO] Portable Project Root: {PROJECT_ROOT}")
ASSET_DIR = os.path.join(PROJECT_ROOT, "assets")

##
# Scene definition
##

@staticmethod
def fov_to_focal_length(fov_degrees, horizontal_aperture=20.955):
    fov_rad = math.radians(fov_degrees)
    return horizontal_aperture / (2 * math.tan(fov_rad / 2))

@configclass
class PreciseManipSceneCfg(InteractiveSceneCfg):
    """Configuration for the UR5e setup with string and needle."""

    # Ground plane & dome light
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
        )

    # -- Robot: UR5e --
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/robots/ur5e_with_gripper.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": 0.0,
                "elbow_joint": 0.0,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
            },
            pos=(0.0, 0.35, 0.51),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                "shoulder_pan_joint", 
                "shoulder_lift_joint", 
                "elbow_joint", 
                "wrist_1_joint", 
                "wrist_2_joint", 
                "wrist_3_joint"
            ],
                stiffness=400.0, damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=300.0, damping=10.0,
            ),
            "passive_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_inner_finger_joint", "left_inner_knuckle_joint", 
                    "right_inner_knuckle_joint", "right_outer_knuckle_joint", 
                    "right_inner_finger_joint"
                ],
                stiffness=0.0, damping=1.0, # No motor force
            ),
        },
    )
    

    # -- Camera: Eye-in-Hand --    
    wrist_cam_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/rs_d435_rgb",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.9299999475479126,
            focus_distance=0.5,
            f_stop=0.0,
            horizontal_aperture=3.8959999084472656,
            vertical_aperture=2.453000068664551,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.03),
            rot=(0.25882, 0.96593, 0, 0),
            convention="opengl",
        ),
        height=1080, width=1936,
        data_types=["rgb"],
        update_period=0.033,
    )
    wrist_cam_depth = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/rs_d435_depth",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.9299999475479126,
            focus_distance=0.6000000238418579,
            f_stop=0.0,
            horizontal_aperture=3.8959999084472656,
            vertical_aperture=2.453000068664551,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.03),
            rot=(0.25882, 0.96593, 0.0, 0.0),
            convention="opengl",
        ),
        height=1080, width=1936,
        update_period=0.033,
        data_types=["depth"],
    )

    # -- Camera: External --
    external_cam_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ext_rs_d435_rgb",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.9299999475479126,
            focus_distance=0.5,
            f_stop=0.0,
            horizontal_aperture=3.8959999084472656,
            vertical_aperture=2.453000068664551,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.3, 0.35, 1.0),
            rot=(0.61237, 0.35355, 0.35355, 0.61237),
            convention="opengl",
        ),
        height=1080, width=1936,
        data_types=["rgb"],
        update_period=0.033,
    )

    # -- Task Objects --
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/objects/common/riro_table.usd",
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    needle_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/NeedleBase",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/objects/dispo_needle_base.usd",
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0.3, 0.505)),
    )

    needle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Needle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/objects/dispo_needle_3mm.usd",
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, 0.3, 0.54),
            rot=(0.7071, 0.0, 0.0, 0.7071),
            ),
    )
    
    string_thread = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/String",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/objects/dispo_string_2.6mm.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.8, 0.3, 1.0),
            rot=(0.7071, 0.7071, 0.0, 0.0),
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # Control 6 Joints of the UR5e Arm
    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint", 
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ],
        scale=1.0,
        use_default_offset=True, # Actions are relative to the "Stand-by" pose
    )

    # Control Gripper (0.0 = Open, 1.0 = Closed usually, depends on your USD limits)
    gripper_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations fed to the RL policy."""
        
        # Proprioception: Joint Positions and Velocities
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])})
        
        # Gripper State
        gripper_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])})
        gripper_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()



@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Reset Gripper to Open/Closed state
    reset_gripper = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Rewards and Termination Settings - CHANGE AS NEEDED
## 
@configclass
class RewardsCfg:
    # Basic "Alive" reward to prevent crashing before you define real tasks
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class TerminationsCfg:
    # Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class PreciseManipEnvCfg(ManagerBasedRLEnvCfg):
    """Main environment configuration."""
    
    # Link Scene
    scene: PreciseManipSceneCfg = PreciseManipSceneCfg(num_envs=4096, env_spacing=3.0)
    
    # Link MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2  # Control frequency decimation
        self.episode_length_s = 300
        self.viewer.eye = (2.0, 2.0, 2.0) # Camera starting position
        self.sim.dt = 0.005  # Simulation dt
        self.seed = 42  # Random seed
