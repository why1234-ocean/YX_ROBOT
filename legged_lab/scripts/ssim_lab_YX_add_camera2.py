import math
import numpy as np
import mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import mujoco
import yaml
import time
import glfw


# Global camera tracking state
camera_tracking = True

def key_callback(key):
    global camera_tracking

    if key == glfw.KEY_RIGHT:
        cmd[0] = np.min([cmd[0] + 0.1, 0.6])
        print('Increase vx, current vx: {}'.format(cmd[0]))
    elif key == glfw.KEY_LEFT:
        cmd[0] = np.max([cmd[0] - 0.1, -0.8])
        print('Reduce vx, current vx: {}'.format(cmd[0]))

    elif key == glfw.KEY_2:
        cmd[1] = np.min([cmd[1] + 0.1, 0.8])
        print('Increase dyaw, current dyaw: {}'.format(cmd[1]))
    elif key == glfw.KEY_3:
        cmd[1] = np.max([cmd[1] - 0.1, -0.8])
        print('Reduce dyaw, current dyaw: {}'.format(cmd[1]))

    elif key == glfw.KEY_UP:
        cmd[2] = np.min([cmd[2] + 0.05, 2])
        print('Increase dyaw, current dyaw: {}'.format(cmd[2]))
    elif key == glfw.KEY_DOWN:
        cmd[2] = np.max([cmd[2] - 0.1, -2])
        print('Reduce dyaw, current dyaw: {}'.format(cmd[2]))

    elif key == glfw.KEY_C:
        camera_tracking = not camera_tracking
        print(f"Camera tracking: {camera_tracking}")

    elif key == glfw.KEY_1:
        # Front view
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 3.0
        print("Camera: Front view")

    elif key == glfw.KEY_2:
        # Side view
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -10
        viewer.cam.distance = 3.0
        print("Camera: Side view")

    elif key == glfw.KEY_3:
        # Top-down view
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -89
        viewer.cam.distance = 5.0
        print("Camera: Top-down view")

    elif key == glfw.KEY_4:
        # Back view
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10
        viewer.cam.distance = 3.0
        print("Camera: Back view")

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd



def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    w , x, y, z = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

# 运行 ： python deploy/deploy_mujoco/deploy_mujoco.py h1.yaml

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"/home/ocean/下载/YX_ROBOT/legged_lab/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        policy_path = "/home/ocean/下载/YX_ROBOT/logs/YX_new_flat/2026-03-31_10-44-40/exported/policy.pt"

        xml_path = "/home/ocean/下载/YX_ROBOT/legged_lab/resources/YX-1/scene.xml"

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_q_mujoco = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # # Improve simulation stability
    # m.opt.iterations = 100
    # m.opt.ls_iterations = 30

    # Initialize robot state
    # Start from default angles to avoid initial large errors
    d.qpos[7:] = default_angles
    d.qvel[6:] = 0.0
    d.qvel[:6] = 0.0

    # Forward kinematics to initialize everything
    mujoco.mj_forward(m, d)

    # load policy
    policy = torch.jit.load(policy_path)



    # ['left_yaw_joint', 
    # 'right_yaw_joint', 
    # 'left_roll_joint', 
    # 'right_roll_joint', 
    # 'left_pitch_joint', 
    # 'right_pitch_joint', 
    # 'left_knee_joint', 
    # 'right_knee_joint', 
    # 'left_ankle_joint', 
    # 'right_ankle_joint']


    joint_index_mujoco = [ 0, 1, 2, 3, 4 ,   5, 6, 7, 8, 9]
    joint_index_lab = [0, 2, 4, 6, 8,        1, 3, 5, 7, 9]
    
    
    
    last_action = np.zeros((10), dtype=np.double)

    
    hist_obs = deque()
    for _ in range(1):
        hist_obs.append(np.zeros([1, 39], dtype=np.double))

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        # Configure camera to track the robot
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.8]

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_q_mujoco, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            # Clip torque to actuator limits
            tau = np.clip(tau, -335, 335)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1

                                                                                                                            
                                                    
            if counter % control_decimation == 0:
                # Apply control signal here.
                obs = np.zeros([1, 39], dtype=np.float32)

                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                # create observation
                q_lab = np.zeros_like(qj)
                dq_lab = np.zeros_like(dqj)
                q_default = np.zeros_like(qj)


                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                
                    
                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                
                for i in range(10):
                    q_lab[joint_index_lab[i]] = qj[joint_index_mujoco[i]]
                    dq_lab[joint_index_lab[i]] = dqj[joint_index_mujoco[i]]
                
                
                obs[0,0:3] = omega
                obs[0,3:6] = gravity_orientation
                obs[0,6:9] = cmd * cmd_scale

                obs[0,9:19] = q_lab # get_list_action(qj) # qj
                obs[0,19:29] = dq_lab # get_list_action(dqj) # dqj
                obs[0,29:39] = last_action # get_list_action(action)
                
                obs = np.clip(obs, -18, 18)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, 1*39], dtype=np.float32)
                for i in range(1):
                    policy_input[0, i * 39 : (i + 1) *39] = hist_obs[i][0, :]
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -18., 18)
                last_action = action

                # transform action to target_dof_pos
                # target_dof_pos = action * action_scale + default_angles
                for i in range(10):
                    q_default[joint_index_lab[i]] = default_angles[joint_index_mujoco[i]]
                    
                target_dof_pos = action * action_scale + q_default







                target_q_mujoco = np.zeros_like(target_dof_pos)

                for i in range(10):
                    target_q_mujoco[joint_index_mujoco[i]] = target_dof_pos[joint_index_lab[i]]


            # Update camera to follow the robot's base position (if tracking enabled)
            if camera_tracking:
                base_pos = d.qpos[0:3]
                viewer.cam.lookat[:] = base_pos

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)




