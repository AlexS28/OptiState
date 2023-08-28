# This file evaluates the Kalman filter
import copy
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
    mocap_list_2 = cur_traj['mocap_list']
    time_list = cur_traj['time_list']
    traj_length = len(p_list_ref)

    KF = Kalman_Filter()
    x_start = mocap_list[0]
    KF.x[:] = x_start


    model_data = []
    ground_truth_data = []
    measurement_data = []

    time = []
    time.append(0)

    for i in range(traj_length-1):
        # ground truth data from mocap
        ground_truth = mocap_list[i]
        ground_truth_t1 = mocap_list[i+1]

        # initiate the x for propagating the model
        KF.x = ground_truth.reshape(12,1)
        p = p_list_est[i].reshape(12,1)
        x_ref = mocap_list[i].reshape(12,1)
        contact_ref = contact_list[i].reshape(4,1)
        if i != traj_length-1:
            time.append(time[-1]+time_list[i+1]-time_list[i])
        KF.predict_mpc(p,x_ref,contact_ref)

        contact_ref = contact_list[i+1].reshape(4, 1)
        p_cur = p_list_est[i+1]
        dp_cur = dp_list[i+1]
        imu = imu_list[i+1][0:6]
        odom = KF.get_odom(p_cur,dp_cur,contact_ref,imu)
        KF.set_measurements(imu, odom)



        ground_truth_data.append(ground_truth_t1)
        measurement_data.append(KF.z)
        model_data.append(KF.x_model)

# Assuming model_data and ground_truth_data are your lists of lists containing 12x1 numpy arrays
# Convert the lists of lists into a single numpy array for each dataset
# Estimate the innovation (residual) between predictions and measurements
innovation_model = []
for i in range(len(model_data)):
    innovation_model.append(ground_truth_data[i].reshape(12,1) - model_data[i].reshape(12,1))
innovation_array = np.array(innovation_model)
num_samples = innovation_array.shape[0]
innovation_array_2d = innovation_array.reshape(num_samples, -1)

# Calculate the process noise covariance matrix Q
Q = np.var(innovation_array_2d, axis=0)
Q = np.diag(Q)

innovation_meas = []
for i in range(len(measurement_data)):
    cur_meas = measurement_data[i]
    cur_gth = ground_truth_data[i]
    cur_ground_truth = np.array([cur_gth[0][0],cur_gth[1][0],cur_gth[2][0],cur_gth[5][0],cur_gth[6][0],cur_gth[7][0],cur_gth[8][0],cur_gth[9][0],cur_gth[10][0],cur_gth[11][0]]).reshape(10,1)
    innovation_meas.append(cur_ground_truth.reshape(10,1) - cur_meas.reshape(10,1))
innovation_array = np.array(innovation_meas)
num_samples = innovation_array.shape[0]
innovation_array_2d = innovation_array.reshape(num_samples, -1)

# Calculate the process noise covariance matrix Q
R = np.var(innovation_array_2d, axis=0)
# Create a diagonal matrix with the diagonal values
R = np.diag(R)

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
    mocap_list_2 = cur_traj['mocap_list']
    time_list = cur_traj['time_list']
    traj_length = len(p_list_ref)
    KF2 = Kalman_Filter()
    x_start = mocap_list[0]
    KF2.x[:] = x_start
    KF2.Q = Q
    KF2.R = R
    KF2.R[0,0] = 0.0001
    KF2.R[1,1] = 0.0001
    KF2.R[2,2] = 0.0001
    KF2.P = copy.deepcopy(Q)

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
    K_gain = []
    for i in range(traj_length):
        p = p_list_est[i].reshape(12,1)
        dp = dp_list[i].reshape(12,1)
        imu = imu_list[i][0:6].reshape(6,1)
        contact_ref = contact_list[i].reshape(4,1)
        x_ref = mocap_list[i].reshape(12, 1)
        x = KF2.estimate_state_mpc(imu,p,dp,x_ref,contact_ref)
        p_trace.append(KF2.P_trace)
        K_gain.append(KF2.K_gain)

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
                            imu_list[i][6][0], imu_list[i][7][0], imu_list[i][8][0], imu_list[i][9][0],
                            imu_list[i][10][0], imu_list[i][11][0],
                            KF2.f[0, 0], KF2.f[1, 0], KF2.f[2, 0], KF2.f[3, 0], KF2.f[4, 0], KF2.f[5, 0],
                            KF2.f[6, 0], KF2.f[7, 0], KF2.f[8, 0], KF2.f[9, 0], KF2.f[10, 0], KF2.f[11, 0],
                            p[0, 0], p[1, 0], p[2, 0], p[3, 0], p[4, 0], p[5, 0], p[6, 0], p[7, 0], p[8, 0], p[9, 0],
                            p[10, 0], p[11, 0],
                            dp[0, 0], dp[1, 0], dp[2, 0], dp[3, 0], dp[4, 0], dp[5, 0], dp[6, 0], dp[7, 0], dp[8, 0],
                            dp[9, 0], dp[10, 0], dp[11, 0],
                            imu[0, 0], imu[1, 0], imu[2, 0], imu[3, 0], imu[4, 0], imu[5, 0],
                            contact_ref[0, 0], contact_ref[1, 0], contact_ref[2, 0], contact_ref[3, 0]])

        state_MOCAP.append([thx_mocap[i],thy_mocap[i],thz_mocap[i],rx_mocap[i],ry_mocap[i],rz_mocap[i],dthx_mocap[i],dthy_mocap[i],dthz_mocap[i],drx_mocap[i],dry_mocap[i],drz_mocap[i]])
        state_T265.append([thx_t265[i],thy_t265[i],thz_t265[i],rx_t265[i],ry_t265[i],rz_t265[i],dthx_t265[i],dthy_t265[i],dthz_t265[i],drx_t265[i],dry_t265[i],drz_t265[i]])


    plt.figure(1)
    plt.plot(time,dthx_est)
    plt.plot(time,dthx_mocap)
    plt.plot(time,dthx_t265)
    plt.legend(['est','vicon', 't265'])
    plt.title('dthx')

    plt.figure(2)
    plt.plot(time,dthy_est)
    plt.plot(time,dthy_mocap)
    plt.plot(time,dthy_t265)
    plt.legend(['est','vicon', 't265'])
    plt.title('dthy')

    plt.figure(3)
    plt.plot(time,dthz_est)
    plt.plot(time,dthz_mocap)
    plt.plot(time, dthz_t265)
    plt.legend(['est','vicon', 't265'])
    plt.title('dthz')

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
    plt.plot(time,drx_est)
    plt.plot(time,drx_mocap)
    plt.plot(time,drx_t265)
    plt.plot(time,dry_est)
    plt.plot(time,dry_mocap)
    plt.plot(time,dry_t265)
    plt.plot(time,drz_est)
    plt.plot(time,drz_mocap)
    plt.plot(time,drz_t265)
    plt.legend(['est dx','vicon dx','t265 dx','est dy', 'vicon dy', 't265 dy', 'est dz', 'vicon dz', 't265 dz',])
    plt.title('velocity')


    plt.figure(6)
    #plt.plot(time, thx_est)
    #plt.plot(time, thx_mocap)
    #plt.plot(time, thx_t265)
    #plt.plot(time, thy_est)
    #plt.plot(time, thy_mocap)
    #plt.plot(time, thy_t265)
    plt.plot(time, thx_est)
    plt.plot(time, thx_mocap)
    plt.plot(time, thx_t265)
    #plt.legend(['est thx', 'vicon thx', 't265 thx', 'est thy', 'vicon thy', 't265 thy', 'est thz', 'vicon thz', 't265 thz', ])
    plt.legend(['est thz', 'vicon thz', 't265 thz', ])
    plt.title('velocity')


    plt.figure(7)
    plt.plot(p_trace)

    plt.figure(8)
    plt.plot(K_gain)

    if INITIAL_PARAMS.VISUALIZE_DATA_CONVERSION:
        plt.show()

    # save the Kalman filter estimates to the
    # save data in pickle file
    data_collection_dict.update({k+1: {'state_INPUT': state_INPUT, 'state_MOCAP': state_MOCAP, 'state_T265': state_T265}})

    with open(dir_path+'/data_collection/trajectories/rnn_data.pkl', 'wb') as f:
        pickle.dump(data_collection_dict, f)