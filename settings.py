import numpy as np
class INITIAL_PARAMS:
    # OptiState settings
    # Discretized time of data collection
    DT = 0.01
    # Robot height, defined from ground to CoM of in meters
    ROBOT_HEIGHT = 0.28
    # Mass of the robot in kg
    ROBOT_MASS = 8.8
    # Kalman filter frequency in Hz
    KF_FREQUENCY = 500
    # Define the rotational inertia matrix of the robot in the CoM body frame (xx, yy, zz)
    Px = 55303643.08 / (10 ** 9)
    Py = 60119440.34 / (10 ** 9)
    Pz = 105304340.05 / (10 ** 9)
    INERTIA_ROT = np.array([[Px, 0, 0], [0, Py, 0], [0, 0, Pz]])
    # Initialize the starting position of the CoM
    STARTING_STATE = np.array([0.0,0.0,0.0,0.0,0.0,ROBOT_HEIGHT,0.0,0.0,0.0,0.0,0.0,0.0]).reshape(12,1)
    # Initialize the Kalman filter Q and R matrix (prediction and measurement noise respectively)
    # Q noise: thx, thy, thz, x, y, z, dthx, dthy, dthz, dx, dy, dz
    Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.0001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0001])
    # R noise: thx^imu, thy^imu, thz^imu, z^odom, dthx^imu, dthy^imu, dthz^imu, vx^odom, vy^odom, vz^odom
    R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    P = Q