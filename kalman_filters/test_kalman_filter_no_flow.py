import pickle
import matplotlib.pyplot as plt
import numpy as np

from kalman_filter_no_flow import Kalman_Filter

with open('data_collection/saved_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

thx_vicon = data_collection['thx_vicon']
thy_vicon = data_collection['thy_vicon']
thz_vicon = data_collection['thz_vicon']
rx_vicon = data_collection['rx_vicon']
ry_vicon = data_collection['ry_vicon']
rz_vicon = data_collection['rz_vicon']
drx_vicon = data_collection['drx_vicon']
dry_vicon = data_collection['dry_vicon']
drz_vicon = data_collection['drz_vicon']
dthx_vicon = data_collection['dthx_vicon']
dthy_vicon = data_collection['dthy_vicon']
dthz_vicon = data_collection['dthz_vicon']
thx_imu = data_collection['thx_imu']
thy_imu = data_collection['thy_imu']
thz_imu = data_collection['thz_imu']
dthx_imu = data_collection['dthx_imu']
dthy_imu = data_collection['dthy_imu']
dthz_imu = data_collection['dthz_imu']
x_imu = data_collection['x_imu']
y_imu = data_collection['x_imu']
dx_imu = data_collection['dx_imu']
dy_imu = data_collection['dy_imu']

p1x = data_collection['p1x']
p1y = data_collection['p1y']
p1z = data_collection['p1z']
p2x = data_collection['p2x']
p2y = data_collection['p2y']
p2z = data_collection['p2z']
p3x = data_collection['p3x']
p3y = data_collection['p3y']
p3z = data_collection['p3z']
p4x = data_collection['p4x']
p4y = data_collection['p4y']
p4z = data_collection['p4z']
dp1x = data_collection['dp1x']
dp1y = data_collection['dp1y']
dp1z = data_collection['dp1z']
dp2x = data_collection['dp2x']
dp2y = data_collection['dp2y']
dp2z = data_collection['dp2z']
dp3x = data_collection['dp3x']
dp3y = data_collection['dp3y']
dp3z = data_collection['dp3z']
dp4x = data_collection['dp4x']
dp4y = data_collection['dp4y']
dp4z = data_collection['dp4z']
f1x = data_collection['f1x']
f1y = data_collection['f1y']
f1z = data_collection['f1z']
f2x = data_collection['f2x']
f2y = data_collection['f2y']
f2z = data_collection['f2z']
f3x = data_collection['f3x']
f3y = data_collection['f3y']
f3z = data_collection['f3z']
f4x = data_collection['f4x']
f4y = data_collection['f4y']
f4z = data_collection['f4z']
contact = data_collection['contact']
time = data_collection['time']

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

moving_average_dx = [0.0]*7
moving_average_dy = [0.0]*4

KF = Kalman_Filter()

state_KF = []
state_VICON = []

for i in range(len(time)):
    # we first build our arrays for p (12x1), dp (12x1), f (12x1), imu (6x1), contact (4x1)
    p = np.array([p1x[i],p1y[i],p1z[i],p2x[i],p2y[i],p2z[i],p3x[i],p3y[i],p3z[i],p4x[i],p4y[i],p4z[i]]).reshape(12,1)
    dp = np.array([dp1x[i],dp1y[i],dp1z[i],dp2x[i],dp2y[i],dp2z[i],dp3x[i],dp3y[i],dp3z[i],dp4x[i],dp4y[i],dp4z[i]]).reshape(12,1)
    f = np.array([f1x[i],f1y[i],f1z[i],f2x[i],f2y[i],f2z[i],f3x[i],f3y[i],f3z[i],f4x[i],f4y[i],f4z[i]]).reshape(12,1)
    imu = np.array([thx_imu[i],thy_imu[i],thz_imu[i],dthx_imu[i],dthy_imu[i],dthz_imu[i]]).reshape(6,1)

    cur_contact = contact[i]

    x = KF.estimate_state(imu,None,p,dp,cur_contact,f)

    moving_average_dx.append(x[9])
    moving_average_dy.append(x[10])

    del moving_average_dx[0]
    del moving_average_dy[0]

    x[9] = sum(moving_average_dx)/len(moving_average_dx)
    x[10] = sum(moving_average_dy)/len(moving_average_dy)

    KF.x[9] = x[9]
    KF.x[10] = x[10]

    thx_est.append(x[0])
    thy_est.append(x[1])
    thz_est.append(x[2])
    rx_est.append(x[3])
    ry_est.append(x[4])
    rz_est.append(x[5])
    dthx_est.append(x[6])
    dthy_est.append(x[7])
    dthz_est.append(x[8])
    drx_est.append(x[9])
    dry_est.append(x[10])
    drz_est.append(x[11])

    state_KF.append([x[0][0], x[1][0], x[2][0], x[3][0], x[4][0], x[5][0], x[6][0], x[7][0], x[8][0], x[9][0], x[10][0], x[11][0]])
    state_VICON.append([thx_vicon[i],thy_vicon[i],thz_vicon[i],rx_vicon[i],ry_vicon[i],rz_vicon[i],dthx_vicon[i],dthy_vicon[i],dthz_vicon[i],drx_vicon[i],dry_vicon[i],drz_vicon[i]])


plt.figure(1)
plt.plot(time,dthx_est)
plt.plot(time,dthx_imu)
plt.plot(time,dthx_vicon)
plt.legend(['est','imu','vicon'])
plt.title('thx')

plt.figure(2)
plt.plot(time,dthy_est)
plt.plot(time,dthy_imu)
plt.plot(time,dthy_vicon)
plt.legend(['est','imu','vicon'])
plt.title('thy')

plt.figure(3)
plt.plot(time,dthz_est)
plt.plot(time,dthz_imu)
plt.plot(time,dthz_vicon)
plt.legend(['est','imu','vicon'])
plt.title('thz')

plt.figure(4)
plt.plot(time,rx_est)
plt.plot(time,rx_vicon)
plt.plot(time,ry_est)
plt.plot(time,ry_vicon)
plt.plot(time,rz_est)
plt.plot(time,rz_vicon)
plt.legend(['est x','vicon x', 'est y','vicon y', 'est z','vicon z'])
plt.title('position')

plt.figure(5)
plt.plot(time,drx_est)
plt.plot(time,drx_vicon)
plt.plot(time,dry_est)
plt.plot(time,dry_vicon)
plt.plot(time,drz_est)
plt.plot(time,drz_vicon)
plt.legend(['est dx','vicon dx', 'est dy','vicon dy', 'est dz','vicon dz'])
plt.title('velocity')
plt.show()


# save the Kalman filter estimates to the
# save data in pickle file
data_collection = {}
data_collection.update({'state_KF': state_KF, 'state_VICON': state_VICON})

with open('../data_collection/rnn_data.pkl', 'wb') as f:
    pickle.dump(data_collection, f)





