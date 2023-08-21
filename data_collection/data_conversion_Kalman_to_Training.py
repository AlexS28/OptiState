# This file evaluates the Kalman filter
import sys
import os
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)
import pickle
import matplotlib.pyplot as plt
import numpy as np
from settings import INITIAL_PARAMS
import os
from kalman_filter.kalman_filter import Kalman_Filter

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
traj_num = 0
filter_horizon = INITIAL_PARAMS.FILTER_HORIZON_KALMAN
# ANSI escape code for orange text
orange_color_code = '\033[93m'
reset_color_code = '\033[0m'

with open(dir_path + '/data_collection/trajectories/saved_trajectories.pkl', 'rb') as f:
    data_collection = pickle.load(f)
data_collection_dict = {}

for k in range(len(data_collection)):
    cur_traj = data_collection[k+1]
    print(f"{orange_color_code}'NOW SAVING TRAJECTORY: {k+1}'{reset_color_code}")
    p_list_est = cur_traj['p_list_est']
    p_list_ref = cur_traj['p_list_ref']
    dp_list = cur_traj['dp_list']
    imu_list = cur_traj['imu_list']
    f_list = cur_traj['f_list']
    contact_list = cur_traj['contact_list']
    t265_list = cur_traj['t265_list']
    mocap_list = cur_traj['mocap_list']
    time_list = cur_traj['time_list']
    traj_length = len(p_list_ref)

    KF = Kalman_Filter()
    x_start = mocap_list[0]
    KF.x[:] = x_start

    Q00 = []
    Q11 = []
    Q22 = []
    Q33 = []
    Q44 = []
    Q55 = []
    Q66 = []
    Q77 = []
    Q88 = []
    Q99 = []
    Q1010 = []
    Q1111 = []

    R00 = []
    R11 = []
    R22 = []
    R33 = []
    R44 = []
    R55 = []
    R66 = []
    R77 = []
    R88 = []
    R99 = []
    R1010 = []

    time = []
    time.append(0)

    for i in range(traj_length):
        p = p_list_est[i].reshape(12,1)
        x_ref = mocap_list[i].reshape(12,1)
        contact_ref = contact_list[i].reshape(4,1)
        if i != traj_length-1:
            time.append(time[-1]+time_list[i+1]-time_list[i])
        KF.predict_mpc(p,x_ref,contact_ref)

        p_cur = p_list_est[i]
        dp_cur = dp_list[i]
        imu = imu_list[i][0:6]
        odom = KF.get_odom(p_cur,dp_cur,contact_ref,imu)
        KF.set_measurements(imu, odom)

        # ground truth data from mocap
        ground_truth = mocap_list[i]

        # calculate the variances
        Q00.append((KF.x_model[0][0] - ground_truth[0][0]) ** 2)
        Q11.append((KF.x_model[1][0] - ground_truth[1][0]) ** 2)
        Q22.append((KF.x_model[2][0] - ground_truth[2][0]) ** 2)
        Q33.append((KF.x_model[3][0] - ground_truth[3][0]) ** 2)
        Q44.append((KF.x_model[4][0] - ground_truth[4][0]) ** 2)
        Q55.append((KF.x_model[5][0] - ground_truth[5][0]) ** 2)
        Q66.append((KF.x_model[6][0] - ground_truth[6][0]) ** 2)
        Q77.append((KF.x_model[7][0] - ground_truth[7][0]) ** 2)
        Q88.append((KF.x_model[8][0] - ground_truth[8][0]) ** 2)
        Q99.append((KF.x_model[9][0] - ground_truth[9][0]) ** 2)
        Q1010.append((KF.x_model[10][0] - ground_truth[10][0]) ** 2)
        Q1111.append((KF.x_model[11][0] - ground_truth[11][0]) ** 2)

        R00.append((KF.z[0][0] - ground_truth[0][0]) ** 2)
        R11.append((KF.z[1][0] - ground_truth[1][0]) ** 2)
        R22.append((KF.z[2][0] - ground_truth[2][0]) ** 2)
        R33.append((KF.z[3][0] - ground_truth[5][0]) ** 2)
        R44.append((KF.z[4][0] - ground_truth[6][0]) ** 2)
        R55.append((KF.z[5][0] - ground_truth[7][0]) ** 2)
        R66.append((KF.z[6][0] - ground_truth[8][0]) ** 2)
        R77.append((KF.z[7][0] - ground_truth[9][0]) ** 2)
        R88.append((KF.z[8][0] - ground_truth[10][0]) ** 2)
        R99.append((KF.z[9][0] - ground_truth[11][0]) ** 2)

    KF.Q[0, 0] = sum(Q00) / (len(R00))
    KF.Q[1, 1] = sum(Q11) / (len(R00))
    KF.Q[2, 2] = sum(Q22) / (len(R00))
    KF.Q[3, 3] = sum(Q33) / (len(R00))
    KF.Q[4, 4] = sum(Q44) / (len(R00))
    KF.Q[5, 5] = sum(Q55) / (len(R00))
    KF.Q[6, 6] = sum(Q66) / (len(R00))
    KF.Q[7, 7] = sum(Q77) / (len(R00))
    KF.Q[8, 8] = sum(Q88) / (len(R00))
    KF.Q[9, 9] = sum(Q99) / (len(R00))
    KF.Q[10, 10] = sum(Q1010) / (len(R00))
    KF.Q[11, 11] = sum(Q1111) / (len(R00))

    KF.R[0, 0] = sum(R00) / (len(R00))
    KF.R[1, 1] = sum(R11) / (len(R00))
    KF.R[2, 2] = sum(R22) / (len(R00))
    KF.R[3, 3] = sum(R33) / (len(R00))
    KF.R[4, 4] = sum(R44) / (len(R00))
    KF.R[5, 5] = sum(R55) / (len(R00))
    KF.R[6, 6] = sum(R66) / (len(R00))
    KF.R[7, 7] = sum(R77) / (len(R00))
    KF.R[8, 8] = sum(R88) / (len(R00))
    KF.R[9, 9] = sum(R99) / (len(R00))

    KF2 = Kalman_Filter()
    x_start = mocap_list[0]
    KF2.x[:] = x_start
    KF2.Q = KF.Q
    KF2.R = KF.R
    KF2.P = KF.Q

    # plotting results of Kalman filter
    thx_est = []
    thy_est = []
    thz_est = []
    rx_est = []
    ry_est = []
    rz_est = []
    drx_est = []
    dry_est = []
    drz_est = []
    dthx_est = []
    dthy_est = []
    dthz_est = []

    thx_mocap = []
    thy_mocap = []
    thz_mocap = []
    rx_mocap = []
    ry_mocap = []
    rz_mocap = []
    drx_mocap = []
    dry_mocap = []
    drz_mocap = []
    dthx_mocap = []
    dthy_mocap = []
    dthz_mocap = []

    thx_t265 = []
    thy_t265 = []
    thz_t265 = []
    rx_t265 = []
    ry_t265 = []
    rz_t265 = []
    drx_t265 = []
    dry_t265 = []
    drz_t265 = []
    dthx_t265 = []
    dthy_t265 = []
    dthz_t265 = []


    state_INPUT = []
    state_MOCAP = []
    state_T265 = []
    p_trace = []
    for i in range(traj_length):
        p = p_list_est[i].reshape(12,1)
        dp = dp_list[i].reshape(12,1)
        imu = imu_list[i][0:6].reshape(6,1)
        contact_ref = contact_list[i].reshape(4,1)
        x_ref = mocap_list[i].reshape(12, 1)
        x = KF2.estimate_state_mpc(imu,p,dp,x_ref,contact_ref)
        p_trace.append(KF2.P_trace)


        if i == 0:
            moving_average_dthx = [x[6][0]] * filter_horizon
            moving_average_dthy = [x[7][0]] * filter_horizon
            moving_average_dthz = [x[8][0]] * filter_horizon

            moving_average_dx = [x[9][0]] * filter_horizon
            moving_average_dy = [x[10][0]] * filter_horizon
            moving_average_dz = [x[11][0]] * filter_horizon

        moving_average_dx.append(x[9])
        moving_average_dy.append(x[10])
        moving_average_dz.append(x[11])

        moving_average_dthx.append(x[6])
        moving_average_dthy.append(x[7])
        moving_average_dthz.append(x[8])

        del moving_average_dx[0]
        del moving_average_dy[0]
        del moving_average_dz[0]
        del moving_average_dthx[0]
        del moving_average_dthy[0]
        del moving_average_dthz[0]

        x[6] = sum(moving_average_dthx) / len(moving_average_dthx)
        x[7] = sum(moving_average_dthy) / len(moving_average_dthy)
        x[8] = sum(moving_average_dthz) / len(moving_average_dthz)

        x[9] = sum(moving_average_dx)/len(moving_average_dx)
        x[10] = sum(moving_average_dy)/len(moving_average_dy)
        x[11] = sum(moving_average_dz)/len(moving_average_dz)

        KF2.x[6] = x[6]
        KF2.x[7] = x[7]
        KF2.x[8] = x[8]
        KF2.x[9] = x[9]
        KF2.x[10] = x[10]
        KF2.x[11] = x[11]

        # ground truth data from mocap
        ground_truth = mocap_list[i]
        thx_est.append(x[0][0])
        thy_est.append(x[1][0])
        thz_est.append(x[2][0])
        rx_est.append(x[3][0])
        ry_est.append(x[4][0])
        rz_est.append(x[5][0])
        dthx_est.append(x[6][0])
        dthy_est.append(x[7][0])
        dthz_est.append(x[8][0])
        drx_est.append(x[9][0])
        dry_est.append(x[10][0])
        drz_est.append(x[11][0])
        thx_mocap.append(ground_truth[0][0])
        thy_mocap.append(ground_truth[1][0])
        thz_mocap.append(ground_truth[2][0])
        rx_mocap.append(ground_truth[3][0])
        ry_mocap.append(ground_truth[4][0])
        rz_mocap.append(ground_truth[5][0])
        dthx_mocap.append(ground_truth[6][0])
        dthy_mocap.append(ground_truth[7][0])
        dthz_mocap.append(ground_truth[8][0])
        drx_mocap.append(ground_truth[9][0])
        dry_mocap.append(ground_truth[10][0])
        drz_mocap.append(ground_truth[11][0])
        t265 = t265_list[i]
        thx_t265.append(t265[0][0])
        thy_t265.append(t265[1][0])
        thz_t265.append(t265[2][0])
        rx_t265.append(t265[3][0])
        ry_t265.append(t265[4][0])
        rz_t265.append(t265[5][0])
        dthx_t265.append(t265[6][0])
        dthy_t265.append(t265[7][0])
        dthz_t265.append(t265[8][0])
        drx_t265.append(t265[9][0])
        dry_t265.append(t265[10][0])
        drz_t265.append(t265[11][0])

        state_INPUT.append([x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0], x[6][0], x[7][0], x[8][0], x[9][0], x[10][0], x[11][0],
                            imu_list[i][6][0], imu_list[i][7][0], imu_list[i][8][0], imu_list[i][9][0], imu_list[i][10][0], imu_list[i][11][0],
                            KF2.f[0,0], KF2.f[1,0], KF2.f[2,0], KF2.f[3,0], KF2.f[4,0], KF2.f[5,0],
                            KF2.f[6,0], KF2.f[7,0], KF2.f[8,0], KF2.f[9,0], KF2.f[10,0], KF2.f[11,0]])

        state_MOCAP.append([thx_mocap[i],thy_mocap[i],thz_mocap[i],rx_mocap[i],ry_mocap[i],rz_mocap[i],dthx_mocap[i],dthy_mocap[i],dthz_mocap[i],drx_mocap[i],dry_mocap[i],drz_mocap[i]])
        state_T265.append([thx_t265[i],thy_t265[i],thz_t265[i],rx_t265[i],ry_t265[i],rz_t265[i],dthx_t265[i],dthy_t265[i],dthz_t265[i],drx_t265[i],dry_t265[i],drz_t265[i]])


    plt.figure(1)
    plt.plot(time,dthx_est)
    plt.plot(time,dthx_mocap)
    plt.plot(time,dthx_t265)
    plt.legend(['est','vicon', 't265'])
    plt.title('thx')

    plt.figure(2)
    plt.plot(time,dthy_est)
    plt.plot(time,dthy_mocap)
    plt.plot(time,dthy_t265)
    plt.legend(['est','vicon', 't265'])
    plt.title('thy')

    plt.figure(3)
    plt.plot(time,dthz_est)
    plt.plot(time,dthz_mocap)
    plt.plot(time, dthz_t265)
    plt.legend(['est','vicon', 't265'])
    plt.title('thz')

    plt.figure(4)
    plt.plot(time,rx_est)
    plt.plot(time,rx_mocap)
    plt.plot(time, rx_t265)
    plt.plot(time,ry_est)
    plt.plot(time,ry_mocap)
    plt.plot(time, ry_t265)
    plt.plot(time,rz_est)
    plt.plot(time,rz_mocap)
    plt.plot(time, rz_t265)
    plt.legend(['est x','vicon x','t265 x', 'est y','vicon y', 't265 y','est z','vicon z','t265 z'])
    plt.title('position')

    plt.figure(5)
    #plt.plot(time,drx_est)
    #plt.plot(time,drx_mocap)
    #plt.plot(time,dry_est)
    #plt.plot(time,dry_mocap)
    plt.plot(time,drz_est)
    plt.plot(time,drz_mocap)
    plt.legend(['est dx','vicon dx','est dy', 'vicon dy', 'est dz', 'vicon dz'])
    plt.title('velocity')

    plt.figure(6)
    plt.plot(p_trace)
    if INITIAL_PARAMS.VISUALIZE_DATA_CONVERSION:
        plt.show()

    # save the Kalman filter estimates to the
    # save data in pickle file
    data_collection_dict.update({k+1: {'state_INPUT': state_INPUT, 'state_MOCAP': state_MOCAP, 'state_T265': state_T265}})

    with open(dir_path+'/data_collection/trajectories/rnn_data.pkl', 'wb') as f:
        pickle.dump(data_collection_dict, f)