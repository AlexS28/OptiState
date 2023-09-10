import copy
import numpy as np
import scaler_kin.v3.SCALER_v2_Leg_6DOF_gripper
import scipy.io
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2
import pickle
import sys
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)
from settings import INITIAL_PARAMS

# Necessary info: imu, p, dp, contact, f, t265, mocap
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
folder_path = dir_path + '/OptiState/data_collection/trajectories/'
file_list = os.listdir(folder_path)
file_list = sorted(file_list)
data_collection = {}
# Create a directory to save images
combined_image_directory = dir_path + f'/OptiState/data_collection/trajectories/saved_images/saved_images_combined'
if not os.path.exists(combined_image_directory):
    os.makedirs(combined_image_directory)
# moving average filter applied to velocity components
filter_horizon_t265 = INITIAL_PARAMS.FILTER_HORIZON_T265
filter_horizon_mocap = INITIAL_PARAMS.FILTER_HORIZON_MOCAP
cutoff = INITIAL_PARAMS.DATA_CUTOFF_START
end_cutoff = copy.deepcopy(INITIAL_PARAMS.DATA_CUTOFF_END)
dt = INITIAL_PARAMS.DT

# ANSI escape code for orange text
orange_color_code = '\033[93m'
reset_color_code = '\033[0m'
traj_num = 1
image_iteration = 1
for file_name in file_list:
    if file_name.endswith(".mat"):
        print(f"{orange_color_code}'NOW SAVING TRAJECTORY: {traj_num}'{reset_color_code}")
        file_path = os.path.join(folder_path, file_name)

        if traj_num >= 11 and traj_num < 16:
            end_cutoff = copy.deepcopy(INITIAL_PARAMS.DATA_CUTOFF_END_2)
        else:
            end_cutoff = copy.deepcopy(INITIAL_PARAMS.DATA_CUTOFF_END)
        data = scipy.io.loadmat(file_path)

        # we first plot to double check
        data_p_est = data['foot_state_history']
        data_p_ref = data['footSteps_ref']
        data_bodyCM_ref = data['bodyCM_ref']
        data_bodyR_ref = data['bodyR_ref']
        data_forces = data['control_history']
        data_liftLeg = data['liftLeg_ref']
        data_t265 = data['body_state_history']
        data_time = data['time_history']
        data_imu = data['imu']
        data_encoder = data['encoder_history']
        data_depth = data['depth4']
        data_mocap = data['mocap_history']

        # initialize list of variables used in the Kalman filter
        p_list_est = []
        p_list_ref = []
        dp_list = []
        f_list = []
        contact_list = []
        imu_list = []
        t265_list = []
        mocap_list = []
        depth_list = []
        time_list = []
        ref_list = []

        def quaternion_2_euler(quaternion):
            """ quaternion 2 euler
                This method converts quaterninon to euler angles (X-Y-Z "fixed" angle)
                The Euler angle definition follows equation 2.64 in the following text book, "Introduction to Robotics Mechanics and Control Third Edition, John J. Craig"
                http://www.mech.sharif.ir/c/document_library/get_file?uuid=5a4bb247-1430-4e46-942c-d692dead831f&groupId=14040
                Args:
                    quaternion: Orientation represented as a quaternion vector of form q = [w, x, y, z], with w as the scalar
                                number. [dim: 1 x 4]
                Returns:
                    euler_angle [degree as a unit]: rotation along "fixed" x, y, z [dim: 3 x 1]. This is extrinsic rotation
                    So, if you want to consider rotation matrix,
                    R_{from A to B} = R_{along z-axis \alpha} * R_{along y-axis \beta} * R_{along x-axis \gamma}
                """
            xyzw_quat = np.array([quaternion[1],quaternion[2],quaternion[3],quaternion[0]])
            r = R.from_quat(xyzw_quat)
            euler_angle = r.as_euler('xyz')
            euler = np.array([euler_angle[0],euler_angle[1],euler_angle[2]]).reshape(3,1)

            return euler

        # mocap plot
        rx_mocap_plot = []
        ry_mocap_plot = []
        rz_mocap_plot = []

        drx_mocap_plot = []
        dry_mocap_plot = []
        drz_mocap_plot = []

        thx_mocap_plot = []
        thy_mocap_plot = []
        thz_mocap_plot = []

        dthx_mocap_plot = []
        dthy_mocap_plot = []
        dthz_mocap_plot = []

        # we first need to modify the mocap data so it's initialized to 0,0,0 at the beginning
        initial_mocap_quat = data_mocap[0,3:]
        initial_mocap_rot = R.from_quat(initial_mocap_quat)
        initial_mocap_pos = data_mocap[0,0:3]/1000.0

        initial_mocap_pos_correct = copy.deepcopy(initial_mocap_pos)
        initial_mocap_R = R.from_quat(initial_mocap_quat)

        initial_t265_quat = R.from_quat(np.array([0.0,0.0,0.0,1.0]))
        initial_mocap_relative = initial_t265_quat.inv()*initial_mocap_R

        T_mocap_t265 = np.zeros((4,4))
        T_mocap_t265[3,3] = 1.0
        T_mocap_t265[0:3,0:3] = initial_mocap_relative.as_matrix()
        T_mocap_t265[0:3,3] = initial_mocap_pos_correct

        for i in range(cutoff-1,data_mocap.shape[0]-1):
            t0 = data_time[0, i]
            t1 = data_time[0, i + 1]
            dt = t1 - t0

            cur_mocap_pos = data_mocap[i,0:3]/1000.0 #- initial_mocap_pos_correct
            cur_mocap_quat = data_mocap[i,3:]

            if np.all(cur_mocap_pos==0) or np.all(cur_mocap_quat==0):
                try:
                    cur_mocap_pos = data_mocap[i-1, 0:3] / 1000.0  # - initial_mocap_pos_correct
                    cur_mocap_quat = data_mocap[i-1, 3:]
                    cur_mocap_R = R.from_quat(cur_mocap_quat)
                    cur_mocap_relative = initial_mocap_rot.inv() * cur_mocap_R
                    data_mocap[i, 3:] = cur_mocap_relative.as_quat()
                except:
                    print('EXCEPTION REACHED, MOCAP DATA INCORRECT')
                    for m in range(i, len(data_mocap)):
                        if np.all(cur_mocap_pos != 0) or np.all(cur_mocap_quat != 0):
                            cur_mocap_pos = data_mocap[i, 0:3] / 1000.0  # - initial_mocap_pos_correct
                            cur_mocap_quat = data_mocap[i, 3:]
                            cur_mocap_R = R.from_quat(cur_mocap_quat)
                            cur_mocap_relative = initial_mocap_rot.inv()*cur_mocap_R
                            data_mocap[i,3:] = cur_mocap_relative.as_quat()
            else:
                cur_mocap_R = R.from_quat(cur_mocap_quat)
                cur_mocap_relative = initial_mocap_rot.inv() * cur_mocap_R
                data_mocap[i, 3:] = cur_mocap_relative.as_quat()
            #matrix_relative = cur_mocap_relative.as_matrix()
            #T_mocap_t265[0:3,0:3] = matrix_relative
            cur_mocap_pos = np.matmul(np.linalg.inv(T_mocap_t265),np.array([cur_mocap_pos[0],cur_mocap_pos[1],cur_mocap_pos[2],1]).reshape(4,1))
            cur_mocap_pos[2] = cur_mocap_pos[2]+0.28
            data_mocap[i,0:3] = cur_mocap_pos[0:3].reshape(3,)

            cur_pos_t0 = data_t265[i, 3:6]
            cur_pos_t1 = data_t265[i+1, 3:6]

            cur_vel_x = (cur_pos_t1[0] - cur_pos_t0[0]) / dt
            cur_vel_y = (cur_pos_t1[1] - cur_pos_t0[1]) / dt
            cur_vel_z = (cur_pos_t1[2] - cur_pos_t0[2]) / dt

            cur_th_t0 = data_t265[i, 0:3]
            cur_th_t1 = data_t265[i + 1, 0:3]

            cur_dth_x = (cur_th_t1[0] - cur_th_t0[0]) / dt
            cur_dth_y = (cur_th_t1[1] - cur_th_t0[1]) / dt
            cur_dth_z = (cur_th_t1[2] - cur_th_t0[2]) / dt

            data_t265[i+1,6:9] = np.array([cur_dth_x, cur_dth_y, cur_dth_z]).reshape(3, )
            data_t265[i+1,9:12] = np.array([cur_vel_x,cur_vel_y,cur_vel_z]).reshape(3,)


        for i in range(cutoff-1,end_cutoff-1):
            t0 = data_time[0,i]
            t1 = data_time[0,i+1]
            dt = t1 - t0
            cur_pos_t0 = data_mocap[i,0:3]
            cur_pos_t1 = data_mocap[i+1,0:3]
            cur_quat_t0 = data_mocap[i, 3:]
            cur_quat_t1 = data_mocap[i+1, 3:]
            cur_euler_t0 = quaternion_2_euler([cur_quat_t0[3], cur_quat_t0[0], cur_quat_t0[1], cur_quat_t0[2]])
            cur_euler_t1 = quaternion_2_euler([cur_quat_t1[3], cur_quat_t1[0], cur_quat_t1[1], cur_quat_t1[2]])

            cur_vel_x = (cur_pos_t1[0] - cur_pos_t0[0]) / dt
            cur_vel_y = (cur_pos_t1[1] - cur_pos_t0[1]) / dt
            cur_vel_z = (cur_pos_t1[2] - cur_pos_t0[2]) / dt

            cur_dthx = (cur_euler_t1[0] - cur_euler_t0[0]) / dt
            cur_dthy = (cur_euler_t1[1] - cur_euler_t0[1]) / dt
            cur_dthz = (cur_euler_t1[2] - cur_euler_t0[2]) / dt

            if i == cutoff-1:
                moving_average_thx = [cur_euler_t1[0][0]]*filter_horizon_mocap
                moving_average_thy = [cur_euler_t1[1][0]]*filter_horizon_mocap
                moving_average_thz = [cur_euler_t1[2][0]]*filter_horizon_mocap

                moving_average_dx = [cur_vel_x] * filter_horizon_mocap
                moving_average_dy = [cur_vel_y] * filter_horizon_mocap
                moving_average_dz = [cur_vel_z] * filter_horizon_mocap
                moving_average_dthx = [cur_dthx] * filter_horizon_mocap
                moving_average_dthy = [cur_dthy] * filter_horizon_mocap
                moving_average_dthz = [cur_dthz] * filter_horizon_mocap

            moving_average_dx.append(cur_vel_x)
            moving_average_dy.append(cur_vel_y)
            moving_average_dz.append(cur_vel_z)
            moving_average_dthx.append(cur_dthx)
            moving_average_dthy.append(cur_dthy)
            moving_average_dthz.append(cur_dthz)
            moving_average_thx.append(cur_euler_t1[0][0])
            moving_average_thy.append(cur_euler_t1[1][0])
            moving_average_thz.append(cur_euler_t1[2][0])

            del moving_average_dx[0]
            del moving_average_dy[0]
            del moving_average_dz[0]
            del moving_average_dthx[0]
            del moving_average_dthy[0]
            del moving_average_dthz[0]
            del moving_average_thx[0]
            del moving_average_thy[0]
            del moving_average_thz[0]

            cur_vel_x = sum(moving_average_dx) / len(moving_average_dx)
            cur_vel_y = sum(moving_average_dy) / len(moving_average_dy)
            cur_vel_z = sum(moving_average_dz) / len(moving_average_dz)
            cur_dthx = sum(moving_average_dthx) / len(moving_average_dthx)
            cur_dthy = sum(moving_average_dthy) / len(moving_average_dthy)
            cur_dthz = sum(moving_average_dthz) / len(moving_average_dthz)
            cur_euler_t1[0][0] = sum(moving_average_thx) / len(moving_average_thx)
            cur_euler_t1[1][0] = sum(moving_average_thy) / len(moving_average_thy)
            cur_euler_t1[2][0] = sum(moving_average_thz) / len(moving_average_thz)

            rx_mocap_plot.append(cur_pos_t1[0])
            ry_mocap_plot.append(cur_pos_t1[1])
            rz_mocap_plot.append(cur_pos_t1[2])

            thx_mocap_plot.append(cur_euler_t1[0][0])
            thy_mocap_plot.append(cur_euler_t1[1][0])
            thz_mocap_plot.append(cur_euler_t1[2][0])

            drx_mocap_plot.append(cur_vel_x)
            dry_mocap_plot.append(cur_vel_y)
            drz_mocap_plot.append(cur_vel_z)

            dthx_mocap_plot.append(cur_dthx)
            dthy_mocap_plot.append(cur_dthy)
            dthz_mocap_plot.append(cur_dthz)

            mocap_list.append(np.array([cur_euler_t1[0][0],cur_euler_t1[1][0],cur_euler_t1[2][0],
                                        cur_pos_t1[0],cur_pos_t1[1],cur_pos_t1[2],
                                        cur_dthx[0], cur_dthy[0], cur_dthz[0],
                                        cur_vel_x, cur_vel_y, cur_vel_z
                                        ]).reshape(12,1))

            ref_list.append(np.array([data_bodyR_ref[i,0],data_bodyR_ref[i,1],data_bodyR_ref[i,2],data_bodyCM_ref[i,0],data_bodyCM_ref[i,1],data_bodyCM_ref[i,2],0.0,0.0,0.0,cur_vel_x,cur_vel_y,0.0]).reshape(12,1))

        thx_t265 = []
        thy_t265 = []
        thz_t265 = []
        rx_t265 = []
        ry_t265 = []
        rz_t265 = []
        dthx_t265 = []
        dthy_t265 = []
        dthz_t265 = []
        drx_t265 = []
        dry_t265 = []
        drz_t265 = []

        for i in range(cutoff-1,end_cutoff-1):
            t0 = data_time[0, i]
            t1 = data_time[0, i + 1]
            dt = t1 - t0

            cur_vel_t0 = data_t265[i, 9:12]
            cur_vel_t1 = data_t265[i + 1,9:12]
            cur_ax = (cur_vel_t1[0] - cur_vel_t0[0]) / dt
            cur_ay = (cur_vel_t1[1] - cur_vel_t0[1]) / dt
            cur_az = (cur_vel_t1[2] - cur_vel_t0[2]) / dt

            cur_dth_t0 = data_t265[i, 6:9]
            cur_dth_t1 = data_t265[i + 1, 6:9]
            cur_ddthx = (cur_dth_t1[0] - cur_dth_t0[0]) / dt
            cur_ddthy = (cur_dth_t1[1] - cur_dth_t0[1]) / dt
            cur_ddthz = (cur_dth_t1[2] - cur_dth_t0[2]) / dt

            if i == cutoff-1:
                moving_average_thx = [data_t265[i,0]] * filter_horizon_t265
                moving_average_thy = [data_t265[i,1]] * filter_horizon_t265
                moving_average_thz = [data_t265[i,2]] * filter_horizon_t265
                moving_average_dx = [data_t265[i,9]] * filter_horizon_t265
                moving_average_dy = [data_t265[i,10]] * filter_horizon_t265
                moving_average_dz = [data_t265[i,11]] * filter_horizon_t265
                moving_average_dthx = [data_t265[i,6]] * filter_horizon_t265
                moving_average_dthy = [data_t265[i,7]] * filter_horizon_t265
                moving_average_dthz = [data_t265[i,8]] * filter_horizon_t265

            moving_average_thx.append(data_t265[i,0])
            moving_average_thy.append(data_t265[i,1])
            moving_average_thz.append(data_t265[i,2])
            moving_average_dx.append(data_t265[i,9])
            moving_average_dy.append(data_t265[i,10])
            moving_average_dz.append(data_t265[i,11])
            moving_average_dthx.append(data_t265[i,6])
            moving_average_dthy.append(data_t265[i,7])
            moving_average_dthz.append(data_t265[i,8])

            del moving_average_thx[0]
            del moving_average_thy[0]
            del moving_average_thz[0]
            del moving_average_dx[0]
            del moving_average_dy[0]
            del moving_average_dz[0]
            del moving_average_dthx[0]
            del moving_average_dthy[0]
            del moving_average_dthz[0]

            cur_thx = sum(moving_average_thx) / len(moving_average_thx)
            cur_thy = sum(moving_average_thy) / len(moving_average_thy)
            cur_thz = sum(moving_average_thz) / len(moving_average_thz)
            cur_vel_x = sum(moving_average_dx) / len(moving_average_dx)
            cur_vel_y = sum(moving_average_dy) / len(moving_average_dy)
            cur_vel_z = sum(moving_average_dz) / len(moving_average_dz)
            cur_dthx = sum(moving_average_dthx) / len(moving_average_dthx)
            cur_dthy = sum(moving_average_dthy) / len(moving_average_dthy)
            cur_dthz = sum(moving_average_dthz) / len(moving_average_dthz)

            thx_t265.append(cur_thx)
            thy_t265.append(cur_thy)
            thz_t265.append(cur_thz)
            rx_t265.append(data_t265[i,3])
            ry_t265.append(data_t265[i,4])
            rz_t265.append(data_t265[i,5])
            dthx_t265.append(cur_dthx)
            dthy_t265.append(cur_dthy)
            dthz_t265.append(cur_dthz)
            drx_t265.append(cur_vel_x)
            dry_t265.append(cur_vel_y)
            drz_t265.append(cur_vel_z)
            t265_list.append(np.array([cur_thx,cur_thy,cur_thz,
                                       data_t265[i,3],data_t265[i,4],data_t265[i,5],
                                       cur_dthx,cur_dthy,cur_dthz,
                                       cur_vel_x,cur_vel_y,cur_vel_z]).reshape(12,1))

            time_list.append(data_time[0,i])
            # thx, thy, thz, dthx, dthy, dthz, ddthx, ddthy, ddthz, ddx, ddy, ddz
            imu_list.append(np.array([data_t265[i,0],data_t265[i,1],data_t265[i,2],data_t265[i,6],data_t265[i,7],data_t265[i,8],data_imu[i,3],data_imu[i,4],data_imu[i,5],data_imu[i,0],data_imu[i,1],data_imu[i,2]]).reshape(12,1))

        plt.figure(1)
        plt.plot(thx_t265)
        plt.plot(thy_t265)
        plt.plot(thz_t265)
        plt.plot(thx_mocap_plot)
        plt.plot(thy_mocap_plot)
        plt.plot(thz_mocap_plot)
        plt.legend(['thx 265','thy 265', 'thz 265','thx mocap', 'thy mocap', 'thz mocap'])

        plt.figure(2)
        plt.plot(rx_t265)
        plt.plot(ry_t265)
        plt.plot(rz_t265)
        plt.plot(rx_mocap_plot)
        plt.plot(ry_mocap_plot)
        plt.plot(rz_mocap_plot)
        plt.legend(['rx 265','ry 265', 'rz 265', 'rx mocap', 'ry mocap', 'rz mocap'])

        plt.figure(3)
        plt.plot(dthx_t265)
        plt.plot(dthy_t265)
        plt.plot(dthz_t265)
        plt.plot(dthx_mocap_plot)
        plt.plot(dthy_mocap_plot)
        plt.plot(dthz_mocap_plot)
        plt.legend(['dthx 265','dthy 265', 'dthz 265','dthx mocap', 'dthy mocap', 'dthz mocap'])

        plt.figure(4)
        plt.plot(drx_t265)
        plt.plot(dry_t265)
        plt.plot(drz_t265)
        plt.plot(drx_mocap_plot)
        plt.plot(dry_mocap_plot)
        plt.plot(drz_mocap_plot)
        plt.legend(['drx 265','dry 265', 'drz 265','drx mocap','dry mocap', 'drz mocap'])

        # plotting reference footsteps (should be in body frame)
        p1x_ref_plot = []
        p1y_ref_plot = []
        p1z_ref_plot = []

        p1x_est_plot = []
        p1y_est_plot = []
        p1z_est_plot = []

        p2x_est_plot = []
        p2y_est_plot = []
        p2z_est_plot = []

        p3x_est_plot = []
        p3y_est_plot = []
        p3z_est_plot = []

        p4x_est_plot = []
        p4y_est_plot = []
        p4z_est_plot = []

        for i in range(cutoff,end_cutoff-1):
            p1x_ref_plot.append(data_p_ref[i, 0, 0])
            p1y_ref_plot.append(data_p_ref[i, 0, 1])
            p1z_ref_plot.append(data_p_ref[i, 0, 2])

            p1x_est_plot.append(data_p_est[i, 0, 0])
            p1y_est_plot.append(data_p_est[i, 0, 1])
            p1z_est_plot.append(data_p_est[i, 0, 2])

            p_list_est.append(np.array([data_p_est[i, 0, 0],data_p_est[i, 0, 1],data_p_est[i, 0, 2],
                                    data_p_est[i, 1, 0],data_p_est[i, 1, 1],data_p_est[i, 1, 2],
                                    data_p_est[i, 2, 0],data_p_est[i, 2, 1],data_p_est[i, 2, 2],
                                    data_p_est[i, 3, 0],data_p_est[i, 3, 1],data_p_est[i, 3, 2]]).reshape(12,1))

            p_list_ref.append(np.array([data_p_ref[i, 0, 0],data_p_ref[i, 0, 1],data_p_ref[i, 0, 2],
                                    data_p_ref[i, 1, 0],data_p_ref[i, 1, 1],data_p_ref[i, 1, 2],
                                    data_p_ref[i, 2, 0],data_p_ref[i, 2, 1],data_p_ref[i, 2, 2],
                                    data_p_ref[i, 3, 0],data_p_ref[i, 3, 1],data_p_ref[i, 3, 2]]).reshape(12,1))

        plt.figure(7)
        plt.plot(p1x_ref_plot)
        plt.plot(p1y_ref_plot)
        plt.plot(p1z_ref_plot)
        plt.plot(p1x_est_plot)
        plt.plot(p1y_est_plot)
        plt.plot(p1z_est_plot)
        plt.legend(['p1x ref','p1y ref','p1z ref','p1x est', 'p1y est', 'p1z est'])

        f1x_plot = []
        f1y_plot = []
        f1z_plot = []
        f2x_plot = []
        f2y_plot = []
        f2z_plot = []
        plt.figure(6)
        for i in range(cutoff,end_cutoff-1):
            f1x_plot.append(data_forces[i, 0])
            f1y_plot.append(data_forces[i, 1])
            f1z_plot.append(data_forces[i, 2])
            f2x_plot.append(data_forces[i, 3])
            f2y_plot.append(data_forces[i, 4])
            f2z_plot.append(data_forces[i, 5])

            f_list.append(np.array([data_forces[i, 0], data_forces[i, 1], data_forces[i, 2],
                                    data_forces[i, 3], data_forces[i, 4], data_forces[i, 5],
                                    data_forces[i, 6], data_forces[i, 7], data_forces[i, 8],
                                    data_forces[i, 9], data_forces[i, 10], data_forces[i, 11]]).reshape(12, 1))

        plt.plot(f1x_plot)
        plt.plot(f1y_plot)
        plt.plot(f1z_plot)
        plt.plot(f2x_plot)
        plt.plot(f2y_plot)
        plt.plot(f2z_plot)
        plt.legend(['f1x', 'f1y', 'f1z', 'f2x', 'f2y', 'f2z'])

        plt.figure(8)
        rx_ref = []
        ry_ref = []
        rz_ref = []

        thx_ref = []
        thy_ref = []
        thz_ref = []


        for i in range(cutoff,end_cutoff-1):
            rx_ref.append(data_bodyCM_ref[i,0])
            ry_ref.append(data_bodyCM_ref[i,1])
            rz_ref.append(data_bodyCM_ref[i,2])
            thx_ref.append(data_bodyR_ref[i,0])
            thy_ref.append(data_bodyR_ref[i,1])
            thz_ref.append(data_bodyR_ref[i,2])

        plt.plot(rx_ref)
        plt.plot(ry_ref)
        plt.plot(rz_ref)
        plt.plot(thx_ref)
        plt.plot(thy_ref)
        plt.plot(thz_ref)
        plt.legend(['rx ref', 'ry ref', 'rz ref','thx ref', 'thy ref', 'thz ref'])


        dp1_x_est = []
        dp1_y_est = []
        dp1_z_est = []

        for i in range(cutoff,end_cutoff-1):
            dp_cur = np.zeros((12,1))
            for j in range(4):
                theta_t0 = data_encoder[i,j,:]
                theta_t1 = data_encoder[i+1,j,:]
                t0 = data_time[0,i]
                t1 = data_time[0,i+1]
                dt = t1 - t0
                dtheta = (theta_t1 - theta_t0)/(dt)
                jac = scaler_kin.v3.SCALER_v2_Leg_6DOF_gripper.Leg.leg_jacobian_3DoF(data_encoder[i,j,:],which_leg=j)
                dp_est = np.matmul(jac,dtheta.reshape(3,1))/1000.0
                dp_cur[3*j:3*j+3] = dp_est

            dp1_x_est.append(dp_cur[0])
            dp1_y_est.append(dp_cur[1])
            dp1_z_est.append(dp_cur[2])
            dp_list.append(dp_cur)

        plt.figure(9)
        plt.plot(dp1_x_est)
        plt.plot(dp1_y_est)
        plt.plot(dp1_z_est)
        plt.legend(['dp1 x', 'dp1 y', 'dp1 z'])

        for i in range(cutoff,end_cutoff-1):
            cur_liftLeg = data_liftLeg[i]
            cur_contact = []
            for i in range(4):
                if cur_liftLeg[i] == 0:
                    cur_contact.append(1)
                else:
                    cur_contact.append(0)
            contact_list.append(np.array([cur_contact[0],cur_contact[1],cur_contact[2],cur_contact[3]]).reshape(4,1))

        # Create a directory to save images
        image_directory = dir_path + f'/OptiState/data_collection/trajectories/saved_images/saved_images_traj_{traj_num}'
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

        for i in range(cutoff, end_cutoff-1):
            depth_image_np = data_depth[i]
            # Convert to 8-bit image (assuming depth_image_normalized is in the range [0, 1])
            depth_image_8bit = (depth_image_np * 255).astype(np.uint8)

            # Save the depth_image_1 as PNG file in the created folder with a numbered filename
            file_name = f"{image_directory}/image_{i}.png"
            cv2.imwrite(file_name, depth_image_8bit)
            file_name_combined = f"{combined_image_directory}/image_{image_iteration}.png"
            cv2.imwrite(file_name_combined, depth_image_8bit)
            image_iteration += 1

        if INITIAL_PARAMS.VISUALIZE_DATA_CONVERSION:
            plt.show()

        # save trajectory into pkl file
        data_collection.update({traj_num:{'p_list_est': p_list_est, 'p_list_ref': p_list_ref, 'dp_list': dp_list, 'imu_list': imu_list, 'f_list': f_list,
                                   'contact_list': contact_list, 't265_list': t265_list, 'mocap_list': mocap_list, 'ref_list': ref_list, 'time_list': time_list}})

        with open(dir_path+'/OptiState/data_collection/trajectories/saved_trajectories.pkl', 'wb') as f:
            pickle.dump(data_collection, f)

        traj_num += 1




        #if traj_num == 3:
        #    break

#depth_image_1 = data_depth[0]
#window_name = f'RealSense {0}'
#cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
#cv2.imshow(window_name, depth_image_1)
#cv2.waitKey(10000)

# Save the depth_image_1
#file_name = 'depth_image_1.png'
#cv2.imwrite(file_name, depth_image_1)