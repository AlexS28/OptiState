import copy
import numpy as np
from settings import INITIAL_PARAMS
from misc.force_controller import StanceController
from misc.force_controller import next_state

class Kalman_Filter:
    def __init__(self):
        # state predicted by KF: thx, thy, thz, x, y, z, dthx, dthy, dthz, dx, dy, dz
        self.x = INITIAL_PARAMS.STARTING_STATE
        # measurement: thx^imu, thy^imu, thz^imu, z^odom, dthx^imu, dthy^imu, dthz^imu,
        # vx^odom, vy^odom, vz^odom
        self.z = np.zeros((10,1))
        # we define the H matrix based on the measurements defined in z
        self.H = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0,0,0,0],
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
        self.R = INITIAL_PARAMS.R
        self.P_trace = np.trace(self.P)
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
        self.dt = INITIAL_PARAMS.DT

        # model state
        self.x_model = INITIAL_PARAMS.STARTING_STATE

        Q = np.array([100.0, 100.0, 100.0, 0.0, 0.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        Q = np.diag(Q)
        R_VALUE = 0.00001
        R = np.array(
            [R_VALUE, R_VALUE, R_VALUE, R_VALUE, R_VALUE, R_VALUE, R_VALUE, R_VALUE, R_VALUE, R_VALUE, R_VALUE,
             R_VALUE])
        R = np.diag(R)
        P = Q
        # stance controller
        self.stance_controller = StanceController(5, Q, R, P, self.dt)
        self.p_mpc = np.zeros((12, 5 + 1))
        self.body_mpc = np.zeros((12, 5 + 1))
        self.contact_mpc = np.zeros((4, 5))

    def get_odom(self, p_cur, dp_cur, contact_cur, imu):
        sum_contacts = sum(contact_cur)
        base_vel_x, base_vel_y, base_vel_z = 0.0, 0.0, 0.0
        base_pos_z = 0.0
        for i in range(4):
            if contact_cur[i] == 1:
                base_vel_x = base_vel_x + dp_cur[3 * i]
                base_vel_y = base_vel_y + dp_cur[3 * i + 1]
                base_pos_z = base_pos_z + p_cur[3 * i + 2]
            if contact_cur[i] == 0:
                base_vel_z = base_vel_z + dp_cur[3 * i + 2]

        if not sum_contacts == 0:
            base_vel_x = -1 * base_vel_x / sum_contacts
            base_vel_y = -1 * base_vel_y / sum_contacts
            base_vel_z = -1 * base_vel_z / sum_contacts
            base_pos_z = -1 * base_pos_z / sum_contacts
            base_velocity = np.array([base_vel_x, base_vel_y, base_vel_z]).reshape(3, 1)
        else:
            base_velocity = np.array([0.0, 0.0, 0.0]).reshape(3, 1)

        # transform into world frame
        R_base_to_world = self.rotation_matrix_body_world(imu[0], imu[1], imu[2])
        base_velocity[:, 0] = np.matmul(R_base_to_world, base_velocity).reshape(3, )
        odom = np.array([base_pos_z, base_velocity[0], base_velocity[1], base_velocity[2]]).reshape(4,1)

        return odom


    def set_measurements(self, imu, odom):
        # thx ^ imu, thy ^ imu, thz ^ imu, z ^ odom,  dthx ^ imu, dthy ^ imu, dthz ^ imu,
        # vx^odom, vy^odom, vz^odom
        # imu is a 6x1 array: thx, thy, thz, dthx, dthy, dthz
        self.z[0:3] = imu[0:3]
        self.z[4:7] = imu[3:6]

        # odom is a 4x1 array: z, vx, vy, vz
        self.z[3] = odom[0]
        self.z[7:10] = odom[1:]

    def predict(self, p, f):
        # this function predicts the next state x and covariance P based on the current state and current covariance P
        # inputs are p: 12x1 array (x,y,z) position per leg | f: 12x1 array(x,y,z) ground reaction force per leg
        # we first must build the state space matrix F and B
        # we calculate the rotation matrix from the body frame to the world frame
        R = self.rotation_matrix_body_world(self.x[0], self.x[1], self.x[2])
        self.F[0:3, 6:9] = np.transpose(R)

        # calculate the discretized version of F and B matrices
        self.F_d = self.identity_large + self.dt * self.F
        self.B_d = self.dt * self.B

        # we now calculate the prediction step using our model-based equation
        #self.x = np.matmul(self.F_d, self.x) + np.matmul(self.B_d, f) + self.dt * self.g
        self.x = next_state(self.x, p, f, self.dt)

        self.P = np.matmul(np.matmul(self.F_d, self.P), np.transpose(self.F_d)) + self.Q

        self.x_model = copy.deepcopy(self.x)
        self.P_trace = copy.deepcopy(np.trace(self.P))

    def predict_mpc(self, p, body_ref, cur_contact):
        self.p_mpc[:,0] = p.reshape(12,)
        self.p_mpc[:,1:] = p
        self.body_mpc[:,0] = self.x.reshape(12,)
        self.body_mpc[:,1:] = body_ref
        self.contact_mpc[:,0] = cur_contact.reshape(4,)
        self.contact_mpc[:,1:] = cur_contact
        self.stance_controller.opti.set_value(self.stance_controller.p_mpc, self.p_mpc)
        self.stance_controller.opti.set_value(self.stance_controller.body_mpc, self.body_mpc)
        self.stance_controller.opti.set_value(self.stance_controller.contact_mpc, self.contact_mpc)
        sol = self.stance_controller.opti.solve()
        # retrieve desired reaction forces (control output of the MPC)
        f = sol.value(self.stance_controller.controls)

        R = self.rotation_matrix_body_world(self.x[0], self.x[1], self.x[2])
        self.F[0:3, 6:9] = np.transpose(R)

        # calculate the discretized version of F and B matrices
        self.F_d = self.identity_large + self.dt * self.F

        self.P = np.matmul(np.matmul(self.F_d, self.P), np.transpose(self.F_d)) + self.Q

        self.x = next_state(self.x, p, f[:,0].reshape(12,1), self.dt)
        self.x_model = copy.deepcopy(self.x)
        self.P_trace = copy.deepcopy(np.trace(self.P))

    def update(self):
        # this is the update step for our Kalman filter. Before running this function, run the prediction function first
        y_bar = self.z - np.matmul(self.H, self.x)
        S = np.matmul(np.matmul(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.inv(S))
        self.x = self.x + np.matmul(K, y_bar)
        self.P = np.matmul(self.identity_large - np.matmul(K, self.H), self.P)

    def estimate_state(self, imu, p, dp, contact, f):
        # this function will estimate the state of the robot using the Kalman filter
        # inputs are the current measurements:
        # imu is a 6x1 array: thx, thy, thz, dthx, dthy, dthz
        # odom is a 4x1 array: z, vx, vy, vz
        # flow is a 3x1 array: dthz, vx, vy
        # lidar is a 1x1 array: z
        # p: 12x1 array (x,y,z) position per leg
        # f: 12x1 array(x,y,z) ground reaction force per leg
        # output: update state x
        odom = self.get_odom(p,dp,contact,imu)
        self.set_measurements(imu, odom)
        self.predict(p,f)
        self.update()

        return self.x

    def estimate_state_mpc(self, imu, p, dp, body_ref, contact):
        odom = self.get_odom(p, dp, contact, imu)
        self.set_measurements(imu, odom)
        self.predict_mpc(p, body_ref, contact)
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
        return np.array([[0, -x[2][0], x[1][0]],
                         [x[2][0], 0, -x[0][0]],
                         [-x[1][0], x[0][0], 0]])



if __name__ == '__main__':
    KF = Kalman_Filter()