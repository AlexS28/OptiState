# This file evaluates the Kalman filter
import pickle
import matplotlib.pyplot as plt
import numpy as np
from settings import INITIAL_PARAMS
import os
from kalman_filter.kalman_filter import Kalman_Filter

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
traj_num = 0
filter_horizon = 1
with open(dir_path + '/OptiState/data_collection/trajectories/saved_trajectories.pkl', 'rb') as f:
    data_collection = pickle.load(f)

cur_traj = data_collection[1]

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
#KF2.Q = KF.Q
#KF2.R = KF.R
#KF2.P = KF.Q

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


state_KF = []
state_MOCAP = []
for i in range(traj_length):
    p = p_list_est[i].reshape(12,1)
    dp = dp_list[i].reshape(12,1)
    imu = imu_list[i][0:6].reshape(6,1)
    contact_ref = contact_list[i].reshape(4,1)
    x_ref = mocap_list[i].reshape(12, 1)
    x = KF2.estimate_state_mpc(imu,p,dp,x_ref,contact_ref)

    if i == 0:
        moving_average_dx = [x[9][0]] * filter_horizon
        moving_average_dy = [x[10][0]] * filter_horizon
        moving_average_dz = [x[11][0]] * filter_horizon

    moving_average_dx.append(x[9])
    moving_average_dy.append(x[10])
    moving_average_dz.append(x[11])

    del moving_average_dx[0]
    del moving_average_dy[0]
    del moving_average_dz[0]

    x[9] = sum(moving_average_dx)/len(moving_average_dx)
    x[10] = sum(moving_average_dy)/len(moving_average_dy)
    x[11] = sum(moving_average_dz)/len(moving_average_dz)
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
    drx_mocap.append(ground_truth[6][0])
    dry_mocap.append(ground_truth[7][0])
    drz_mocap.append(ground_truth[8][0])
    dthx_mocap.append(ground_truth[9][0])
    dthy_mocap.append(ground_truth[10][0])
    dthz_mocap.append(ground_truth[11][0])

    state_KF.append([x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0], x[6][0], x[7][0], x[8][0], x[9][0], x[10][0], x[11][0]])
    state_MOCAP.append([thx_mocap[i],thy_mocap[i],thz_mocap[i],rx_mocap[i],ry_mocap[i],rz_mocap[i],dthx_mocap[i],dthy_mocap[i],dthz_mocap[i],drx_mocap[i],dry_mocap[i],drz_mocap[i]])


#plt.figure(1)
#plt.plot(time,thx_est)
#plt.plot(time,thx_mocap)
#plt.legend(['est','vicon'])
#plt.title('thx')

#plt.figure(2)
#plt.plot(time,thy_est)
#plt.plot(time,thy_mocap)
#plt.legend(['est','vicon'])
#plt.title('thy')

#plt.figure(3)
#plt.plot(time,thz_est)
#plt.plot(time,thz_mocap)
#plt.legend(['est','vicon'])
#plt.title('thz')

plt.figure(4)
plt.plot(time,rx_est)
plt.plot(time,rx_mocap)
plt.plot(time,ry_est)
plt.plot(time,ry_mocap)
plt.plot(time,rz_est)
plt.plot(time,rz_mocap)
plt.legend(['est x','vicon x', 'est y','vicon y', 'est z','vicon z'])
plt.title('position')

plt.figure(5)
plt.plot(time,drx_est)
plt.plot(time,drx_mocap)
#plt.plot(time,dry_est)
#plt.plot(time,dry_mocap)
#plt.plot(time,drz_est)
#plt.plot(time,drz_mocap)
plt.legend(['est dx','vicon dx'])
#plt.title('velocity')
plt.show()