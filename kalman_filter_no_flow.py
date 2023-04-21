import numpy as np
from settings import INITIAL_PARAMS

class Kalman_Filter:
    def __init__(self):
        # state predicted by KF: thx, thy, thz, x, y, z, dthx, dthy, dthz, dx, dy, dz
        self.x = INITIAL_PARAMS.STARTING_STATE
        # measurement w/o factor graph: thx^imu, thy^imu, thz^imu, z^odom, z^lidar, dthx^imu, dthy^imu, dthz^imu,
        # vx^odom, vy^odom, vz^odom
        self.z = np.zeros((11,1))
        # we define the H matrix based on the measurements defined in z
        self.H = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,1]])
        # we define the initial parameters for Q and R
        # Kalman Filter parameters
        self.P = INITIAL_PARAMS.P
        self.Q = INITIAL_PARAMS.Q
        self.R = INITIAL_PARAMS.R_NO_FLOW
        # we now define our model parameters
        # robot mass
        self.m = INITIAL_PARAMS.ROBOT_MASS
        # rotational inertia matrix
        self.inertia_rot = INITIAL_PARAMS.INERTIA_ROT
        # we define some helper matrices for the model prediction step
        self.zero_mat = np.array([[0,0,0],[0,0,0],[0,0,0]])
        # identity matrix
        self.identity = np.eye(3, 3)
        # large identity matrix
        self.identity_large = np.eye(12, 12)
        # identity over mass matrix
        self.identity_m = np.array([[1 / self.m, 0, 0], [0, 1 / self.m, 0], [0, 0, 1 / self.m]])
        # setup A, and B model prediction matrices
        self.F = np.vstack((np.hstack((self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat)),
                            np.hstack((self.zero_mat, self.zero_mat, self.zero_mat, self.identity)),
                            np.hstack((self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat)),
                            np.hstack((self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat))))

        self.B = np.vstack((np.hstack((self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat)),
                            np.hstack((self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat)),
                            np.hstack((self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat)),
                            np.hstack((self.identity_m, self.identity_m, self.identity_m, self.identity_m))))

        # gravity term
        self.g = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -9.81]).reshape(12, 1)

        # we calculate the discretized time of our model equation
        self.dt = 1.0/INITIAL_PARAMS.KF_FREQUENCY

    def set_measurements(self, imu, odom, lidar):
        # thx ^ imu, thy ^ imu, thz ^ imu, z ^ odom, z ^ lidar, dthx ^ imu, dthy ^ imu, dthz ^ imu,
        # vx^odom, vy^odom, vz^odom

        # imu is a 6x1 array: thx, thy, thz, dthx, dthy, dthz
        self.z[0:3] = imu[0:3]
        self.z[5:8] = imu[3:6]

        # odom is a 4x1 array: z, vx, vy, vz
        self.z[3] = odom[0]
        self.z[8:11] = odom[1:]

        # lidar is a 1x1 array: z
        self.z[4] = lidar[0]

    def predict(self, p, f):
        # this function predicts the next state x and covariance P based on the current state and current covariance P
        # inputs are p: 12x1 array (x,y,z) position per leg | f: 12x1 array(x,y,z) ground reaction force per leg
        # we first must build the state space matrix F and B
        # we calculate the rotation matrix from the body frame to the world frame
        R = self.rotation_matrix_body_world(self.x[0], self.x[1], self.x[2])
        self.F[0:3, 6:9] = np.transpose(R)
        I_hat_intertia_rot = np.matmul(R, np.matmul(self.inertia_rot, np.transpose(R)))
        I_hat_inv_inertia_rot = np.linalg.inv(I_hat_intertia_rot)

        p[0:3] = np.matmul(R, p[0:3].reshape(3, 1)).reshape(3, 1)
        p[3:6] = np.matmul(R, p[3:6].reshape(3, 1)).reshape(3, 1)
        p[6:9] = np.matmul(R, p[6:9].reshape(3, 1)).reshape(3, 1)
        p[9:12] = np.matmul(R, p[9:12].reshape(3, 1)).reshape(3, 1)

        d1 = self.skew(np.vstack((p[0, 0], p[1, 0], p[2, 0])))
        d2 = self.skew(np.vstack((p[3, 0], p[4, 0], p[5, 0])))
        d3 = self.skew(np.vstack((p[6, 0], p[7, 0], p[8, 0])))
        d4 = self.skew(np.vstack((p[9, 0], p[10, 0], p[11, 0])))

        self.B[6:9, 0:3] = np.matmul(I_hat_inv_inertia_rot, d1)
        self.B[6:9, 3:6] = np.matmul(I_hat_inv_inertia_rot, d2)
        self.B[6:9, 6:9] = np.matmul(I_hat_inv_inertia_rot, d3)
        self.B[6:9, 9:] = np.matmul(I_hat_inv_inertia_rot, d4)

        # calculate the discretized version of F and B matrices
        self.F_d = self.identity_large + self.dt * self.F
        self.B_d = self.dt * self.B

        # we now calculate the prediction step using our model-based equation
        self.x = np.matmul(self.F_d, self.x) + np.matmul(self.B_d, f) + self.dt * self.g
        self.P = np.matmul(np.matmul(self.F_d, self.P), np.transpose(self.F_d)) + self.Q

    def update(self):
        # this is the update step for our Kalman filter. Before running this function, run the prediction function first
        y_bar = self.z - np.matmul(self.H, self.x)
        S = np.matmul(np.matmul(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.inv(S))
        self.x = self.x + np.matmul(K, y_bar)
        self.P = np.matmul(self.identity_large - np.matmul(K, self.H), self.P)

    def estimate_state(self, imu, odom, lidar, p, f):
        # this function will estimate the state of the robot using the Kalman filter
        # inputs are the current measurements:
        # imu is a 6x1 array: thx, thy, thz, dthx, dthy, dthz
        # odom is a 4x1 array: z, vx, vy, vz
        # flow is a 3x1 array: dthz, vx, vy
        # lidar is a 1x1 array: z
        # p: 12x1 array (x,y,z) position per leg
        # f: 12x1 array(x,y,z) ground reaction force per leg
        # output: update state x
        self.set_measurements(imu, odom, lidar)
        self.predict(p,f)
        self.update()

        return self.x

    def rotation_matrix_body_world(self, thx, thy, thz):
        # rotation matrix of body to world
        th = np.array([thx, thy, thz])
        Rz = np.vstack((np.hstack((np.cos(th[2]), -np.sin(th[2]), 0)), np.hstack((np.sin(th[2]), np.cos(th[2]), 0)), np.hstack((0, 0, 1))))
        Ry = np.vstack((np.hstack((np.cos(th[1]), 0, np.sin(th[1]))), np.hstack((0, 1, 0)), np.hstack((-np.sin(th[1]), 0, np.cos(th[1])))))
        Rx = np.vstack((np.hstack((1, 0, 0)), np.hstack((0, np.cos(th[0]), -np.sin(th[0]))), np.hstack((0, np.sin(th[0]), np.cos(th[0])))))

        R = np.matmul(Rz, np.matmul(Ry, Rx))

        return R

    def skew(self, x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])



if __name__ == '__main__':
    KF = Kalman_Filter()