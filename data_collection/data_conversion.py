# this file converts data from vicon and estimate trajectories into a dictionary for state estimation
from scaler_kin.v3.SCALAR_kinematics import scalar_k
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

scaler_k_obj = scalar_k()
from scaler_kin import Leg
leg = Leg()


def rotation_matrix_body_world(thx, thy, thz, x, y, z):
    w_T_b = np.zeros((4,4))
    # rotation matrix of body to world
    th = np.array([thx, thy, thz])
    Rz = np.vstack(
        (np.hstack((np.cos(th[2]), -np.sin(th[2]), 0)), np.hstack((np.sin(th[2]), np.cos(th[2]), 0)), np.hstack((0, 0, 1))))
    Ry = np.vstack(
        (np.hstack((np.cos(th[1]), 0, np.sin(th[1]))), np.hstack((0, 1, 0)), np.hstack((-np.sin(th[1]), 0, np.cos(th[1])))))
    Rx = np.vstack(
        (np.hstack((1, 0, 0)), np.hstack((0, np.cos(th[0]), -np.sin(th[0]))), np.hstack((0, np.sin(th[0]), np.cos(th[0])))))

    R = np.transpose(np.matmul(Rz, np.matmul(Ry, Rx)))
    w_T_b[0:3,0:3] = R
    w_T_b[0,3] = x
    w_T_b[1,3] = y
    w_T_b[2,3] = z
    w_T_b[3,3] = 1.0

    return w_T_b

def ft_sensor_to_world(w_T_b, wrench_sensor_frame, joint_angles, leg):
    b_T_toe = scaler_k_obj.scalar_forward_kinematics_3DoF(which_leg=leg, joint_angles=joint_angles[0:3], with_body=True, body_angle = 0.0, output_xyz = False)
    w_T_toe = np.dot(w_T_b, b_T_toe)
    w_R_toe = w_T_toe[0:3, 0:3]
    rotx90 = np.array([[1,               0,                0],
                       [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                       [0, np.sin(np.pi/2),  np.cos(np.pi/2)]])
    roty90 = np.array([[ np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                        [               0, 1,               0],
                        [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])

    w_R_ft = np.dot(np.dot(w_R_toe, rotx90), roty90)

    f_w = np.dot(w_R_ft, wrench_sensor_frame)

    return f_w


# moving average filter applied to velocity components
moving_average_history = 5
indx_cutoff1 = 383 # starting point of robot motion in vicon
indx_cutoff2 = 750 # ending point of robot motion in robot state
force_threshold = 7

estdata_robot = scipy.io.loadmat("estimated_traj/Estimated_traj23.mat")
encoder_val = estdata_robot['enc_val']
encoder_val_vel = estdata_robot['enc_vel_val']
imu_ang_rate = estdata_robot['imu_ang_rate']
imu_acc = estdata_robot['imu_accel']
time = estdata_robot['time']
wrench = estdata_robot['wrench']
ref_CoM = estdata_robot['ref_val']

imu_x = []
x_plot =[]
dx_plot = []
x = 0.0
dx = 0.0
for i in range(imu_acc.shape[0]):
    dx = dx + 0.5 * imu_acc[i, 0] * 0.01
    x = x + dx*0.01 + 0.5*imu_acc[i,0]*0.01**2
    x_plot.append(x)
    dx_plot.append(dx)

#plt.plot(dx_plot)



#plt.plot(imu_acc[:,1])
#plt.plot(imu_acc[:,2])
#plt.show()

# footstep positions
p1_list, p2_list, p3_list, p4_list = [], [], [], []
dp1_list, dp2_list, dp3_list, dp4_list = [], [], [], []
# footstep forces
f1_list, f2_list, f3_list, f4_list = [], [], [], []
# contact list
contact_list = []

# IMU
theta_CoM_IMU = []
dtheta_CoM_IMU = []
pos_CoM_IMU = []
dpos_CoM_IMU = []
acc_CoM_IMU = []
theta_CoM_IMU_cur_x = 0.0
theta_CoM_IMU_cur_y = 0.0
theta_CoM_IMU_cur_z = 0.0
time_history = []
time_history.append(0.0)
theta_CoM_IMU.append([0.0,0.0,0.0])
dtheta_CoM_IMU.append([0.0,0.0,0.0])
pos_CoM_IMU.append([0.0,0.0,0.0])
dpos_CoM_IMU.append([0.0,0.0,0.0])
acc_CoM_IMU.append([0.0,0.0,0.0])

wrench0 = wrench[0,:]
theta_CoM_IMU_0 = theta_CoM_IMU[0]
w_T_b = rotation_matrix_body_world(theta_CoM_IMU_0[0],theta_CoM_IMU_0[1],theta_CoM_IMU_0[2],0,0,0)
f1_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0],wrench0[1],wrench0[2]]).reshape(3,1),encoder_val[0,3*0:3*0+3],0)
f2_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0+6],wrench0[1+6],wrench0[2+6]]).reshape(3,1),encoder_val[0,3*1:3*1+3],1)
f3_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0+6*2],wrench0[1+6*2],wrench0[2+6*2]]).reshape(3,1),encoder_val[0,3*2:3*2+3],2)
f4_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0+6*3],wrench0[1+6*3],wrench0[2+6*3]]).reshape(3,1),encoder_val[0,3*3:3*3+3],3)

f1_list.append([f1_0[0][0],f1_0[1][0],f1_0[2][0]])
f2_list.append([f2_0[0][0],f2_0[1][0],f2_0[2][0]])
f3_list.append([f3_0[0][0],f3_0[1][0],f3_0[2][0]])
f4_list.append([f4_0[0][0],f4_0[1][0],f4_0[2][0]])

contact_list.append([1,1,1,1])

x_pos_IMU = 0.0
y_pos_IMU = 0.0
z_pos_IMU = 0.0
dx_pos_IMU = 0.0
dy_pos_IMU = 0.0
dz_pos_IMU = 0.0

# get footstep positions and velocities from encoder
for k in range(4):
    cur_encoder = encoder_val[0, 3*k:3*k+3]
    cur_footstep = scaler_k_obj.scalar_forward_kinematics_3DoF(k,cur_encoder,with_body=True,output_xyz=True)

    cur_encoder_vel = encoder_val_vel[0, 3*k:3*k+3]
    cur_footstep_vel = np.matmul(leg.leg_jacobian_3DoF(cur_encoder,k),cur_encoder_vel.reshape(3,1))

    if k == 0:
        p1_list.append(list(cur_footstep/1000))
        dp1_list.append(list(cur_footstep_vel/1000))
    if k == 1:
        p2_list.append(list(cur_footstep/1000))
        dp2_list.append(list(cur_footstep_vel/1000))
    if k == 2:
        p3_list.append(list(cur_footstep/1000))
        dp3_list.append(list(cur_footstep_vel/1000))
    if k == 3:
        p4_list.append(list(cur_footstep/1000))
        dp4_list.append(list(cur_footstep_vel/1000))


for i in range(1,len(encoder_val)-1):
    time_history.append(time[0,i])
    # get dt from Time
    dt0 = time[0,i]
    dt1 = time[0,i+1]
    dt = dt1 - dt0

    # get angular rate from IMU
    imu_ang_rate_cur = list(imu_ang_rate[i,:])
    dtheta_CoM_IMU.append(imu_ang_rate_cur)
    theta_CoM_IMU_cur_x = theta_CoM_IMU_cur_x + dt*imu_ang_rate_cur[0]
    theta_CoM_IMU_cur_y = theta_CoM_IMU_cur_y + dt*imu_ang_rate_cur[1]
    theta_CoM_IMU_cur_z = theta_CoM_IMU_cur_z + dt*imu_ang_rate_cur[2]
    theta_CoM_IMU.append([theta_CoM_IMU_cur_x, theta_CoM_IMU_cur_y, theta_CoM_IMU_cur_z])

    x_pos_IMU = x_pos_IMU + dx_pos_IMU + 0.5*imu_acc[i,0]*dt**2
    y_pos_IMU = y_pos_IMU + dy_pos_IMU + 0.5*imu_acc[i,1]*dt**2
    z_pos_IMU = z_pos_IMU + dz_pos_IMU + 0.5*(-imu_acc[i,2]-9.81)*dt**2
    dx_pos_IMU = dx_pos_IMU + 0.5*imu_acc[i,0]*dt
    dy_pos_IMU = dy_pos_IMU + 0.5*imu_acc[i,1]*dt
    dz_pos_IMU = dz_pos_IMU + 0.5*(-imu_acc[i,2]-9.81)*dt

    # velocity IMU
    pos_CoM_IMU.append([x_pos_IMU,y_pos_IMU,z_pos_IMU])
    dpos_CoM_IMU.append([dx_pos_IMU,dy_pos_IMU,dz_pos_IMU])
    acc_CoM_IMU.append([imu_acc[i,0],imu_acc[i,1],-imu_acc[i,2]-9.81])


    wrench0 = wrench[i, :]

    w_T_b = rotation_matrix_body_world(theta_CoM_IMU_cur_x, theta_CoM_IMU_cur_y, theta_CoM_IMU_cur_z, 0, 0, 0)
    f1_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0],wrench0[1],wrench0[2]]).reshape(3,1),encoder_val[i,3*0:3*0+3],0)
    f2_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0+6],wrench0[1+6],wrench0[2+6]]).reshape(3,1),encoder_val[i,3*1:3*1+3],1)
    f3_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0+6*2],wrench0[1+6*2],wrench0[2+6*2]]).reshape(3,1),encoder_val[i,3*2:3*2+3],2)
    f4_0 = ft_sensor_to_world(w_T_b,np.array([wrench0[0+6*3],wrench0[1+6*3],wrench0[2+6*3]]).reshape(3,1),encoder_val[i,3*3:3*3+3],3)

    f1_list.append([f1_0[0][0], f1_0[1][0], f1_0[2][0]])
    f2_list.append([f2_0[0][0], f2_0[1][0], f2_0[2][0]])
    f3_list.append([f3_0[0][0], f3_0[1][0], f3_0[2][0]])
    f4_list.append([f4_0[0][0], f4_0[1][0], f4_0[2][0]])

    if f1_0[2] >= force_threshold:
        contact_list1 = 1
    else:
        contact_list1 = 0

    if f2_0[2] >= force_threshold:
        contact_list2 = 1
    else:
        contact_list2 = 0

    if f3_0[2] >= force_threshold:
        contact_list3 = 1
    else:
        contact_list3 = 0

    if f4_0[2] >= force_threshold:
        contact_list4 = 1
    else:
        contact_list4 = 0

    contact_list.append([contact_list1, contact_list2, contact_list3, contact_list4])

    # get footstep positions from encoder
    for k in range(4):
        cur_encoder = encoder_val[i, 3*k:3*k+3]
        cur_footstep = scaler_k_obj.scalar_forward_kinematics_3DoF(k,cur_encoder,with_body=True,output_xyz=True)

        cur_encoder_vel = encoder_val_vel[i, 3 * k:3 * k + 3]
        cur_footstep_vel = np.matmul(leg.leg_jacobian_3DoF(cur_encoder, k), cur_encoder_vel.reshape(3, 1))

        if k == 0:
            p1_list.append(list(cur_footstep/1000))
            dp1_list.append(list(cur_footstep_vel / 1000))
        if k == 1:
            p2_list.append(list(cur_footstep/1000))
            dp2_list.append(list(cur_footstep_vel / 1000))
        if k == 2:
            p3_list.append(list(cur_footstep/1000))
            dp3_list.append(list(cur_footstep_vel / 1000))
        if k == 3:
            p4_list.append(list(cur_footstep/1000))
            dp4_list.append(list(cur_footstep_vel / 1000))


estdata_vicon = scipy.io.loadmat("vicon/vicon_data23.mat")
vicon_data = estdata_vicon['object_pos']
vicon_time = estdata_vicon['time']


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

vicon_theta = []
vicon_pos = []
vicon_dtheta = []
vicon_vel = []

vicon_moving_average_velx = [0]*moving_average_history
vicon_moving_average_vely = [0]*moving_average_history
vicon_moving_average_velz = [0]*moving_average_history

vicon_moving_average_dthx = [0]*moving_average_history
vicon_moving_average_dthy = [0]*moving_average_history
vicon_moving_average_dthz = [0]*moving_average_history

cur_pos = vicon_data[0,0:3]/1000.0
cur_quat = vicon_data[0,3:]
cur_euler = quaternion_2_euler([cur_quat[-1],cur_quat[0],cur_quat[1],cur_quat[2]])
vicon_theta.append(cur_euler)
vicon_pos.append(list(cur_pos))
vicon_dtheta.append(np.array([0.0,0.0,0.0]).reshape(3,1))
vicon_vel.append([0.0,0.0,0.0])
vicon_time_history = []
vicon_time_history.append(vicon_time[0,0])
for i in range(1,len(vicon_data)-1):
    vicon_time_history.append(vicon_time[0, i])
    t0 = vicon_time[0,i]
    t1 = vicon_time[0,i+1]
    dt = t1 - t0

    cur_pos_t1 = vicon_data[i,0:3]/1000.0
    cur_quat = vicon_data[i,3:]
    cur_euler_t1 = quaternion_2_euler([cur_quat[-1],cur_quat[0],cur_quat[1],cur_quat[2]])

    cur_vel_x = (cur_pos_t1[0] - cur_pos[0])/dt
    cur_vel_y = (cur_pos_t1[1] - cur_pos[1])/dt
    cur_vel_z = (cur_pos_t1[2] - cur_pos[2])/dt

    vicon_moving_average_velx.append(cur_vel_x)
    vicon_moving_average_vely.append(cur_vel_y)
    vicon_moving_average_velz.append(cur_vel_z)

    del vicon_moving_average_velx[0]
    del vicon_moving_average_vely[0]
    del vicon_moving_average_velz[0]

    cur_vel_x = sum(vicon_moving_average_velx) / len(vicon_moving_average_velx)
    cur_vel_y = sum(vicon_moving_average_vely) / len(vicon_moving_average_vely)
    cur_vel_z = sum(vicon_moving_average_velz) / len(vicon_moving_average_velz)

    cur_dthx = (cur_euler_t1[0] - cur_euler[0])/dt
    cur_dthy = (cur_euler_t1[1] - cur_euler[1])/dt
    cur_dthz = (cur_euler_t1[2] - cur_euler[2])/dt

    vicon_moving_average_dthx.append(cur_dthx)
    vicon_moving_average_dthy.append(cur_dthy)
    vicon_moving_average_dthz.append(cur_dthz)

    del vicon_moving_average_dthx[0]
    del vicon_moving_average_dthy[0]
    del vicon_moving_average_dthz[0]

    cur_dthx = sum(vicon_moving_average_dthx) / len(vicon_moving_average_dthx)
    cur_dthy = sum(vicon_moving_average_dthy) / len(vicon_moving_average_dthy)
    cur_dthz = sum(vicon_moving_average_dthz) / len(vicon_moving_average_dthz)

    cur_dth = np.array([cur_dthx,cur_dthy,cur_dthz]).reshape(3,1)

    vicon_pos.append([cur_pos_t1[0], cur_pos_t1[1], cur_pos_t1[2]])
    vicon_theta.append([cur_euler_t1[0], cur_euler_t1[1], cur_euler_t1[2]])
    vicon_dtheta.append([cur_dth[0], cur_dth[1], cur_dth[2]])
    vicon_vel.append([cur_vel_x, cur_vel_y, cur_vel_z])

    cur_pos = cur_pos_t1
    cur_euler = cur_euler_t1

# we plot just the robot states
p1x_list = []
p1y_list = []
p1z_list = []

p2x_list = []
p2y_list = []
p2z_list = []

p3x_list = []
p3y_list = []
p3z_list = []

p4x_list = []
p4y_list = []
p4z_list = []

dp1x_list = []
dp1y_list = []
dp1z_list = []

dp2x_list = []
dp2y_list = []
dp2z_list = []

dp3x_list = []
dp3y_list = []
dp3z_list = []

dp4x_list = []
dp4y_list = []
dp4z_list = []

f1x_list = []
f1y_list = []
f1z_list = []

f2x_list = []
f2y_list = []
f2z_list = []

f3x_list = []
f3y_list = []
f3z_list = []

f4x_list = []
f4y_list = []
f4z_list = []

thx_vicon = []
thy_vicon = []
thz_vicon = []

thx_imu = []
thy_imu = []
thz_imu = []

x_imu = []
y_imu = []
z_imu = []

dthx_imu = []
dthy_imu = []
dthz_imu = []

dx_imu = []
dy_imu = []
dz_imu = []

acc_x_imu = []
acc_y_imu = []
acc_z_imu = []

rx_vicon = []
ry_vicon = []
rz_vicon = []

drx_vicon = []
dry_vicon = []
drz_vicon = []

dthx_vicon = []
dthy_vicon = []
dthz_vicon = []

for i in range(len(p1_list)):
    p1x_list.append(p1_list[i][0])
    p1y_list.append(p1_list[i][1])
    p1z_list.append(p1_list[i][2])

    p2x_list.append(p2_list[i][0])
    p2y_list.append(p2_list[i][1])
    p2z_list.append(p2_list[i][2])

    p3x_list.append(p3_list[i][0])
    p3y_list.append(p3_list[i][1])
    p3z_list.append(p3_list[i][2])

    p4x_list.append(p4_list[i][0])
    p4y_list.append(p4_list[i][1])
    p4z_list.append(p4_list[i][2])

    dp1x_list.append(dp1_list[i][0][0])
    dp1y_list.append(dp1_list[i][1][0])
    dp1z_list.append(dp1_list[i][2][0])

    dp2x_list.append(dp2_list[i][0][0])
    dp2y_list.append(dp2_list[i][1][0])
    dp2z_list.append(dp2_list[i][2][0])

    dp3x_list.append(dp3_list[i][0][0])
    dp3y_list.append(dp3_list[i][1][0])
    dp3z_list.append(dp3_list[i][2][0])

    dp4x_list.append(dp4_list[i][0][0])
    dp4y_list.append(dp4_list[i][1][0])
    dp4z_list.append(dp4_list[i][2][0])

    f1x_list.append(f1_list[i][0])
    f1y_list.append(f1_list[i][1])
    f1z_list.append(f1_list[i][2])

    f2x_list.append(f2_list[i][0])
    f2y_list.append(f2_list[i][1])
    f2z_list.append(f2_list[i][2])

    f3x_list.append(f3_list[i][0])
    f3y_list.append(f3_list[i][1])
    f3z_list.append(f3_list[i][2])

    f4x_list.append(f4_list[i][0])
    f4y_list.append(f4_list[i][1])
    f4z_list.append(f4_list[i][2])

    theta = -np.pi
    # Create the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])


    th_cur_imu = np.array([theta_CoM_IMU[i][0], theta_CoM_IMU[i][1], theta_CoM_IMU[i][2]]).reshape(3,1)
    th_cur_imu_body = np.matmul(R, th_cur_imu)

    thx_imu.append(th_cur_imu_body[0])
    thy_imu.append(th_cur_imu_body[1])
    thz_imu.append(th_cur_imu_body[2])

    x_imu.append(pos_CoM_IMU[i][0])
    y_imu.append(pos_CoM_IMU[i][1])
    z_imu.append(pos_CoM_IMU[i][2])


    dth_cur_imu = np.array([dtheta_CoM_IMU[i][0], dtheta_CoM_IMU[i][1], dtheta_CoM_IMU[i][2]]).reshape(3,1)
    dth_cur_imu_body = np.matmul(R, dth_cur_imu)

    dthx_imu.append(dth_cur_imu_body[0])
    dthy_imu.append(dth_cur_imu_body[1])
    dthz_imu.append(dth_cur_imu_body[2])

    dx_imu.append(dpos_CoM_IMU[i][0])
    dy_imu.append(dpos_CoM_IMU[i][1])
    dz_imu.append(dpos_CoM_IMU[i][2])

    acc_x_imu.append(acc_CoM_IMU[i][0])
    acc_y_imu.append(acc_CoM_IMU[i][1])
    acc_z_imu.append(acc_CoM_IMU[i][2])

for i in range(len(vicon_time_history)):
    thx_vicon.append(vicon_theta[i][0][0])
    thy_vicon.append(vicon_theta[i][1][0])
    thz_vicon.append(vicon_theta[i][2][0])
    rx_vicon.append(vicon_pos[i][0])
    ry_vicon.append(vicon_pos[i][1])
    rz_vicon.append(vicon_pos[i][2])
    drx_vicon.append(vicon_vel[i][0])
    dry_vicon.append(vicon_vel[i][1])
    drz_vicon.append(vicon_vel[i][2])
    dthx_vicon.append(vicon_dtheta[i][0][0])
    dthy_vicon.append(vicon_dtheta[i][1][0])
    dthz_vicon.append(vicon_dtheta[i][2][0])
"""
plt.figure(1)
plt.plot(time_history, p2x_list)
plt.plot(time_history, p2y_list)
plt.plot(time_history, p2z_list)
plt.plot(time_history, dp2x_list)
plt.plot(time_history, dp2y_list)
plt.plot(time_history, dp2z_list)


plt.figure(2)
plt.plot(time_history, thx_imu)
plt.plot(time_history, thy_imu)
plt.plot(time_history, thz_imu)

# data plot vicon
plt.figure(3)
plt.plot(vicon_time_history, thx_vicon)
plt.plot(vicon_time_history, thy_vicon)
plt.plot(vicon_time_history, thz_vicon)

plt.figure(4)
plt.plot(vicon_time_history, rx_vicon)
plt.plot(vicon_time_history, ry_vicon)
plt.plot(vicon_time_history, rz_vicon)

plt.figure(5)
plt.plot(vicon_time_history, drx_vicon)
plt.plot(vicon_time_history, dry_vicon)
plt.plot(vicon_time_history, drz_vicon)

plt.figure(6)
plt.plot(vicon_time_history, dthx_vicon)
plt.plot(vicon_time_history, dthy_vicon)
plt.plot(vicon_time_history, dthz_vicon)

plt.show()
"""

offset_time = vicon_time_history[indx_cutoff1]
vicon_time_history = vicon_time_history[indx_cutoff1:]
for i in range(len(vicon_time_history)):
    vicon_time_history[i] = vicon_time_history[i] - offset_time

for i in range(len(time_history)):
    time_history[i] = vicon_time_history[i]

thx_vicon = thx_vicon[indx_cutoff1:]
thy_vicon = thy_vicon[indx_cutoff1:]
thz_vicon = thz_vicon[indx_cutoff1:]
rx_vicon = rx_vicon[indx_cutoff1:]
ry_vicon = ry_vicon[indx_cutoff1:]
rz_vicon = rz_vicon[indx_cutoff1:]
drx_vicon = drx_vicon[indx_cutoff1:]
dry_vicon = dry_vicon[indx_cutoff1:]
drz_vicon = drz_vicon[indx_cutoff1:]
dthx_vicon = dthx_vicon[indx_cutoff1:]
dthy_vicon = dthy_vicon[indx_cutoff1:]
dthz_vicon = dthz_vicon[indx_cutoff1:]

plt.figure(2)
plt.plot(time_history[:indx_cutoff2], thy_vicon[:indx_cutoff2])
#plt.plot(time_history[:indx_cutoff2], thx_vicon[:indx_cutoff2])
#plt.plot(time_history[:indx_cutoff2], dthz_vicon[:indx_cutoff2])

plt.plot(time_history[:indx_cutoff2], thy_imu[:indx_cutoff2])
#plt.plot(time_history[:indx_cutoff2], thx_imu[:indx_cutoff2])
#plt.plot(time_history[:indx_cutoff2], dthz_imu[:indx_cutoff2])

plt.title("Vicon vs IMU")
plt.xlabel("Time (seconds)")
plt.ylabel("Angle (degrees)")

plt.legend(['Y Vicon','Y IMU'])



# save data in pickle file
data_collection = {}
data_collection.update({'thx_vicon': thx_vicon[:indx_cutoff2], 'thy_vicon': thy_vicon[:indx_cutoff2], 'thz_vicon': thz_vicon[:indx_cutoff2],
                        'rx_vicon': rx_vicon[:indx_cutoff2],'ry_vicon': ry_vicon[:indx_cutoff2], 'rz_vicon': rz_vicon[:indx_cutoff2],
                        'drx_vicon': drx_vicon[:indx_cutoff2], 'dry_vicon': dry_vicon[:indx_cutoff2], 'drz_vicon':drz_vicon[:indx_cutoff2],
                        'dthx_vicon': dthx_vicon[:indx_cutoff2], 'dthy_vicon': dthy_vicon[:indx_cutoff2], 'dthz_vicon': dthz_vicon[:indx_cutoff2],
                        'thx_imu': thx_imu[:indx_cutoff2], 'thy_imu': thy_imu[:indx_cutoff2], 'thz_imu': thz_imu[:indx_cutoff2],
                        'dthx_imu': dthx_imu[:indx_cutoff2], 'dthy_imu': dthy_imu[:indx_cutoff2], 'dthz_imu': dthz_imu[:indx_cutoff2],
                        'x_imu': x_imu[:indx_cutoff2], 'y_imu': y_imu[:indx_cutoff2], 'z_imu': z_imu[:indx_cutoff2],
                        'dx_imu': dx_imu[:indx_cutoff2], 'dy_imu': dy_imu[:indx_cutoff2], 'dz_imu': dz_imu[:indx_cutoff2],
                        'acc_x_imu': acc_x_imu[:indx_cutoff2], 'acc_y_imu': acc_y_imu[:indx_cutoff2], 'acc_z_imu': acc_z_imu[:indx_cutoff2],
                        'p1x': p1x_list[:indx_cutoff2], 'p1y': p1y_list[:indx_cutoff2], 'p1z': p1z_list[:indx_cutoff2],
                        'p2x': p2x_list[:indx_cutoff2], 'p2y': p2y_list[:indx_cutoff2], 'p2z': p2z_list[:indx_cutoff2],
                        'p3x': p3x_list[:indx_cutoff2], 'p3y': p3y_list[:indx_cutoff2], 'p3z': p3z_list[:indx_cutoff2],
                        'p4x': p4x_list[:indx_cutoff2], 'p4y': p4y_list[:indx_cutoff2], 'p4z': p4z_list[:indx_cutoff2],
                        'dp1x': dp1x_list[:indx_cutoff2], 'dp1y': dp1y_list[:indx_cutoff2], 'dp1z': dp1z_list[:indx_cutoff2],
                        'dp2x': dp2x_list[:indx_cutoff2], 'dp2y': dp2y_list[:indx_cutoff2], 'dp2z': p2z_list[:indx_cutoff2],
                        'dp3x': dp3x_list[:indx_cutoff2], 'dp3y': dp3y_list[:indx_cutoff2], 'dp3z': dp3z_list[:indx_cutoff2],
                        'dp4x': dp4x_list[:indx_cutoff2], 'dp4y': dp4y_list[:indx_cutoff2], 'dp4z': p4z_list[:indx_cutoff2],
                        'f1x': f1x_list[:indx_cutoff2], 'f1y': f1y_list[:indx_cutoff2], 'f1z': f1z_list[:indx_cutoff2],
                        'f2x': f2x_list[:indx_cutoff2], 'f2y': f2y_list[:indx_cutoff2], 'f2z': f2z_list[:indx_cutoff2],
                        'f3x': f3x_list[:indx_cutoff2], 'f3y': f3y_list[:indx_cutoff2], 'f3z': f3z_list[:indx_cutoff2],
                        'f4x': f4x_list[:indx_cutoff2], 'f4y': f4y_list[:indx_cutoff2], 'f4z': f4z_list[:indx_cutoff2],
                        'contact': contact_list[:indx_cutoff2],
                        'time': time_history[:indx_cutoff2]})

with open('saved_data.pkl', 'wb') as f:
    pickle.dump(data_collection, f)

plt.show()




