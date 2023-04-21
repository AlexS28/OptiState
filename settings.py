import numpy as np
class INITIAL_PARAMS:
    # OptiState settings
    # Robot height, defined from ground to CoM of in meters
    ROBOT_HEIGHT = 0.32
    # Mass of the robot in kg
    ROBOT_MASS = 0.13
    # Kalman filter frequency in Hz
    KF_FREQUENCY = 500
    # Define the rotational inertia matrix of the robot in the CoM body frame (xx, yy, zz)
    INERTIA_ROT = np.array([[0.02, 0, 0], [0, 0.04, 0], [0, 0, 0.05]])
    # Initialize the starting position of the CoM
    STARTING_STATE = np.array([0.0,0.0,0.0,0.0,0.0,ROBOT_HEIGHT,0.0,0.0,0.0,0.0,0.0,0.0]).reshape(12,1)
    # Initialize the Kalman filter Q and R matrix (prediction and measurement noise respectively)
    # Q noise: thx, thy, thz, x, y, z, dthx, dthy, dthz, dx, dy, dz
    Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # R noise: thx^imu, thy^imu, thz^imu, z^odom, z^lidar, dthx^imu, dthy^imu, dthz^imu, dthz^flow, vx^flow, vy^flow, vx^odom, vy^odom, vz^odom
    R_FLOW = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # R without optical flow: thx^imu, thy^imu, thz^imu, z^odom, z^lidar, dthx^imu, dthy^imu, dthz^imu, vx^odom, vy^odom, vz^odom
    R_NO_FLOW = np.diag([R_FLOW[0,0],R_FLOW[1,1],R_FLOW[2,2],R_FLOW[3,3],R_FLOW[4,4],R_FLOW[5,5],R_FLOW[6,6],R_FLOW[7,7],R_FLOW[11,11],R_FLOW[12,12],R_FLOW[13,13]])
    # R fused noise: thx, thy, thz, z, dthx, dthy, dthz, dx, dy, dz
    R_FUSED = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # R noise for factor graph measurements [z^odom, z^lidar,dthz^imu, dthz^flow,vx^flow, vx^odom,vy^flow,vy^odom]
    R_FACTOR_GRAPH = [R_FLOW[3,3],R_FLOW[4,4],R_FLOW[7,7],R_FLOW[8,8],R_FLOW[9,9],R_FLOW[11,11],R_FLOW[10,10],R_FLOW[12,12]]
    P = Q