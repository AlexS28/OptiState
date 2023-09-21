import numpy as np
class INITIAL_PARAMS:
    # OptiState settings
    # Train without vision
    USE_VISION = True
    # Discretized time of data collection
    DT = 0.01
    # Robot height, defined from ground to CoM of in meters
    ROBOT_HEIGHT = 0.28
    # Mass of the robot in kg
    ROBOT_MASS = 8.8
    # Kalman filter frequency in Hz
    KF_FREQUENCY = 500
    # Horizon for moving average filter applied to the motion capture data (and t265 if available) velocity components
    FILTER_HORIZON_T265 = 1
    FILTER_HORIZON_MOCAP = 1
    # Horizon for moving average filter applied to the Kalman filter velocity components
    FILTER_HORIZON_KALMAN = 2
    # Define the start and end of cutoff for all collected datasets
    DATA_CUTOFF_START = 445
    DATA_CUTOFF_END = 7400
    DATA_CUTOFF_END_2 = 1500
    # Define whether we want to visualize plots for data conversion scripts
    VISUALIZE_DATA_CONVERSION = True
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