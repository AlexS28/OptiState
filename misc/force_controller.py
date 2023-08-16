""" CONFIDENTIAL (C) Mitsubishi Electric Research Labs(MERL)
% 2021
% Alexander Schperberg
% schperberg@merl.com
% 06/20/22
"""
import pdb

# This file provides a stance leg controller (through force MPC), modeled after the paper: 'Dynamic Locomotion in
# the MIT Cheetah 3 Through Convex Model Predictive Control' by Jared Dicarlo, Patrick M. Wensing, Benjamin Katz,
# Gerardo Bledt, and Sangbae Kim'
from casadi import *
import numpy as np

class StanceController():
    def __init__(self, N, Q, R, P, dt):
        # length of the prediction horizon
        self.N = N
        # state weighing matrix
        self.Q = Q
        # control weighing matrix
        self.R = R
        # terminal weighing matrix
        self.P = P
        # discretized time
        self.dt = dt
        # gravity term
        self.g = vertcat(0, 0, 0, 0,0,0,0,0,0,0,0,-9.81)
        # robot mass
        self.m = 8.8
        # set the rotational inertia matrix of the robot in the body frame
        Px = 55303643.08 / (10 ** 9)
        Py = 60119440.34 / (10 ** 9)
        Pz = 105304340.05 / (10 ** 9)
        self.I = np.array([[Px, 0, 0], [0, Py, 0], [0, 0, Pz]])
        # zero matrix
        self.zero_mat = np.array([[0,0,0],[0,0,0],[0,0,0]])
        # identity matrix
        self.identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # identity over mass matrix
        self.identity_m = np.array([[1 / self.m, 0, 0], [0, 1 / self.m, 0], [0, 0, 1 / self.m]])
        # initialize variables
        self.initialize_variables()

    def initialize_variables(self):
        # initialize the opti stack helper class, using the 'conic' keyword as we will use qpOases
        self.opti = casadi.Opti('conic')
        # initialize the robot states: heading angle (th), center of mass base position (r), angular velocity (dth),
        # linear velocity (dr)
        # initialize the robot controls: ground reaction forces per foot
        self.f1 = self.opti.variable(3, self.N)
        self.f2 = self.opti.variable(3, self.N)
        self.f3 = self.opti.variable(3, self.N)
        self.f4 = self.opti.variable(3, self.N)
        self.controls = vertcat(self.f1, self.f2, self.f3, self.f4)
        # provide a parameter for the reference states (reference must include current value)
        self.body_mpc = self.opti.parameter(12, self.N+1)
        # provide a parameter for the reference footstep positions in the base frame (references must include current value)
        self.p_mpc = self.opti.parameter(12, self.N+1)
        # provide a parameter for the contact matrix (0 means foot is in swing, and 1 means foot is in stance)
        self.contact_mpc = self.opti.parameter(4,self.N)
        # since dt may change, we make it a parameter
        self.dt_param = self.opti.parameter(1, 1)
        self.opti.set_value(self.dt_param,self.dt)
        # set up the MPC problem
        self.objective_function()
        self.control_constraints()
        self.initialize_solver()

    def objective_function(self):
        # calculates the summation of cost from the paper
        self.cost = 0
        state = self.body_mpc[:,0]
        for i in range(self.N):
            # get the orientation angles needed for the rotation matrix from the reference trajectory
            thx = self.body_mpc[0, i]
            thy = self.body_mpc[1, i]
            thz = self.body_mpc[2, i]
            # calculate the unique A matrix using the current orientation
            self.A = self.dyanmic_matrix_A(thx, thy, thz)

            # retrieve the reference footstep positions
            r, p1, p2, p3, p4 = self.body_mpc[3:6, i], self.p_mpc[0:3, i], self.p_mpc[3:6, i], self.p_mpc[6:9,
                                                                                                 i], self.p_mpc[9:12, i]

            # calculate the unique B matrix using the current reference orientation and footstep positions
            self.B = self.dynamic_matrix_B(thx, thy, thz, p1, p2, p3, p4)

            control_t = self.controls[:, i]
            # calculate the next state, using the time-invarient linearly discretized equation
            I = np.eye(12, 12)

            state = mtimes(I + self.A * self.dt_param, state) + mtimes(self.B * self.dt_param, control_t) + self.dt_param * self.g

            state_ref = self.body_mpc[:, i+1]
            control = self.controls[:, i]

            if i == self.N-1:
                self.cost = self.cost + mtimes(mtimes((state - state_ref).T, self.P), state - state_ref)\
                            + \
                            mtimes(mtimes((control).T, self.R), control)
            else:
                self.cost = self.cost + mtimes(mtimes((state - state_ref).T, self.Q), state - state_ref) \
                            + \
                            mtimes(mtimes((control).T, self.R), control)

    def control_constraints(self):
        # we must ensure that the force is zero when the leg is in swing. To do this, we calculate the force selection
        # matrix using the reference contact matrix
        for k in range(self.N):
            # create the stance/swing selection matrix
            # if foot is in swing phase, we want to make sure that the forces will be equal to zero at all times. A zero
            # means foot is currently in swing and not in contact with the ground
            D1 = if_else(self.contact_mpc[0, k] == 0, self.identity, self.zero_mat)
            D2 = if_else(self.contact_mpc[1, k] == 0, self.identity, self.zero_mat)
            D3 = if_else(self.contact_mpc[2, k] == 0, self.identity, self.zero_mat)
            D4 = if_else(self.contact_mpc[3, k] == 0, self.identity, self.zero_mat)
            D = vertcat(horzcat(D1, self.zero_mat, self.zero_mat, self.zero_mat),
                        horzcat(self.zero_mat, D2, self.zero_mat, self.zero_mat),
                        horzcat(self.zero_mat, self.zero_mat, D3, self.zero_mat),
                        horzcat(self.zero_mat, self.zero_mat, self.zero_mat, D4))
            # we now set forces to zero for the feet not on the ground
            self.opti.subject_to(mtimes(D, self.controls[:, k]) == 0)

            # for the feet on the ground, we must impose friction cone constraints. A one indicates food is making contact
            F1 = if_else(self.contact_mpc[0, k] == 1, self.identity, self.zero_mat)
            F2 = if_else(self.contact_mpc[1, k] == 1, self.identity, self.zero_mat)
            F3 = if_else(self.contact_mpc[2, k] == 1, self.identity, self.zero_mat)
            F4 = if_else(self.contact_mpc[3, k] == 1, self.identity, self.zero_mat)
            f1 = mtimes(F1, self.controls[0:3, k])
            f2 = mtimes(F2, self.controls[3:6, k])
            f3 = mtimes(F3, self.controls[6:9, k])
            f4 = mtimes(F4, self.controls[9:12, k])

            # we now make the friction cone constraints (note, that if force of the foot is zero, as done using the
            # if_else statement above for feet in swing, the contact constraints are automatically satisfied for that
            # foot)
            self.contact_constraints(f1)
            self.contact_constraints(f2)
            self.contact_constraints(f3)
            self.contact_constraints(f4)

    def contact_constraints(self, f):
        # input of the function is the reaction force of the leg
        fx = f[0]
        fy = f[1]
        fz = f[2]
        self.opti.subject_to(self.opti.bounded(-0,fz,150))
        # coefficient of friction
        mu = 0.6
        # friction cone constraints
        self.opti.subject_to(self.opti.bounded(-mu * fz, fx, mu * fz))
        self.opti.subject_to(self.opti.bounded(-mu * fz, -fx, mu * fz))
        self.opti.subject_to(self.opti.bounded(-mu * fz, fy, mu * fz))
        self.opti.subject_to(self.opti.bounded(-mu * fz, -fy, mu * fz))


    def rotation_matrix_inv(self, thx, thy, thz):
        # thz is yaw, thy is pitch, and thx is roll
        # we assume non-zero values for roll, pitch, and yaw
        # we assume non-zero roll, pitch, and yaw
        # body to world
        R = vertcat(horzcat(cos(thz) / cos(thy), sin(thz) / cos(thy), 0),
                    horzcat(-sin(thz), cos(thz), 0),
                    horzcat(cos(thz) * tan(thy), sin(thz) * tan(thy), 1))

        return R

    def rotation_matrix_body_world(self, thx, thy, thz):
        th  = [thx,thy,thz]
        # rotation matrix of body to world
        Rz = vertcat(horzcat(cos(th[2]), -sin(th[2]), 0), horzcat(sin(th[2]), cos(th[2]), 0), horzcat(0, 0, 1))
        Ry = vertcat(horzcat(cos(th[1]), 0, sin(th[1])), horzcat(0, 1, 0), horzcat(-sin(th[1]), 0, cos(th[1])))
        Rx = vertcat(horzcat(1, 0, 0), horzcat(0, cos(th[0]), -sin(th[0])), horzcat(0, sin(th[0]), cos(th[0])))

        R = mtimes(Rz, mtimes(Ry, Rx))

        return R

    def dyanmic_matrix_A(self, thx, thy, thz):
        # note, thy and thz are from the reference trajectories (we may also average them like in the paper,
        # but in this code I decided not to) - the matrix was changed from how it is written in the paper,
        # with gravity term, instead, inside the A matrix)
        R = self.rotation_matrix_body_world(thx, thy, thz)

        # equation 16 (gravity term has been incorporated into the A matrix for convenience)
        A = vertcat(horzcat(self.zero_mat, self.zero_mat, transpose(R), self.zero_mat),
                    horzcat(self.zero_mat, self.zero_mat, self.zero_mat, self.identity),
                    horzcat(self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat),
                    horzcat(self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat))

        return A

    def dynamic_matrix_B(self, thx, thy, thz, p1, p2, p3, p4):
        # thy, thz is pitch and yaw for the body to world rotation. I is the inertia matrix in the base/body
        # frame, r is the center of mass position of the robot, p1, p2, p3, and p4 are the footstep positions

        # we calculate the inertia tensor in the world frame, and then invert it (equations 15-16)
        R = self.rotation_matrix_body_world(thx, thy, thz)
        I_hat = mtimes(R, mtimes(self.I, transpose(R)))
        I_hat_inv = inv(I_hat)

        p1 = mtimes(R, p1)
        p2 = mtimes(R, p2)
        p3 = mtimes(R, p3)
        p4 = mtimes(R, p4)

        d1 = skew(vertcat(p1[0], p1[1], p1[2]))
        d2 = skew(vertcat(p2[0], p2[1], p2[2]))
        d3 = skew(vertcat(p3[0], p3[1], p3[2]))
        d4 = skew(vertcat(p4[0], p4[1], p4[2]))

        # calculate matrix B (equation 16)
        B = vertcat(horzcat(self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat),
                    horzcat(self.zero_mat, self.zero_mat, self.zero_mat, self.zero_mat),
                    horzcat(mtimes(I_hat_inv, d1), mtimes(I_hat_inv, d2), mtimes(I_hat_inv, d3), mtimes(I_hat_inv, d4)),
                    horzcat(self.identity_m, self.identity_m, self.identity_m, self.identity_m))

        return B

    def initialize_solver(self):
        # initiate objective function into solver
        opts = {'printLevel': "none"}
        self.opti.minimize(self.cost)
        self.opti.solver('qpoases', opts)

def rotation_matrix_body_world_numpy(thx, thy, thz):
    # thx,thy,thz = thx[0],thy[0],thz[0]
    # rotation matrix of body to world
    th = np.array([thx, thy, thz])
    Rz = np.vstack((np.hstack((np.cos(th[2]), -np.sin(th[2]), 0)), np.hstack((np.sin(th[2]), np.cos(th[2]), 0)), np.hstack((0, 0, 1))))
    Ry = np.vstack((np.hstack((np.cos(th[1]), 0, np.sin(th[1]))), np.hstack((0, 1, 0)), np.hstack((-np.sin(th[1]), 0, np.cos(th[1])))))
    Rx = np.vstack((np.hstack((1, 0, 0)), np.hstack((0, np.cos(th[0]), -np.sin(th[0]))), np.hstack((0, np.sin(th[0]), np.cos(th[0])))))

    R = np.matmul(Rz, np.matmul(Ry, Rx))

    return R

# calculates P Matrix below
zero_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# identity matrix
identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# robot mass
robot_m = 8.8
# identity over mass matrix
identity_m = np.array([[1 / robot_m, 0, 0], [0, 1 / robot_m, 0],
                           [0, 0, 1 / robot_m ]])
A = np.vstack((np.hstack((zero_mat, zero_mat, zero_mat, zero_mat)),
                   np.hstack((zero_mat, zero_mat, zero_mat, identity)),
                   np.hstack((zero_mat, zero_mat, zero_mat, zero_mat)),
                   np.hstack((zero_mat, zero_mat, zero_mat, zero_mat))))
B = np.vstack((np.hstack((zero_mat, zero_mat, zero_mat, zero_mat)),
                   np.hstack((zero_mat, zero_mat, zero_mat, zero_mat)),
                   np.hstack((zero_mat, zero_mat, zero_mat, zero_mat)),
                   np.hstack((identity_m, identity_m, identity_m, identity_m))))
# set the rotational inertia matrix of the robot in the body frame
Px = 55303643.08 / (10 ** 9)
Py = 60119440.34 / (10 ** 9)
Pz = 105304340.05 / (10 ** 9)
I = np.array([[Px, 0, 0], [0, Py, 0], [0, 0, Pz]])
I_identity = np.eye(12, 12)
g = np.array([0, 0, 0, 0,0,0,0,0,0,0,0,-9.81]).reshape(12,1)

def skew_numpy(x):
    return np.array([[0, -x[2][0], x[1][0]],
                    [x[2][0], 0, -x[0][0]],
                    [-x[1][0], x[0][0], 0]])

def next_state(cur_state, cur_footstep, control_t, dt):
    R = rotation_matrix_body_world_numpy(cur_state[0], cur_state[1], cur_state[2])
    A[0:3, 6:9] = np.transpose(R)
    I_hat = np.matmul(R, np.matmul(I, np.transpose(R)))
    I_hat_inv = np.linalg.inv(I_hat)
    cur_footstep[0:3] = np.matmul(R, cur_footstep[0:3].reshape(3, 1)).reshape(3, 1)
    cur_footstep[3:6] = np.matmul(R, cur_footstep[3:6].reshape(3, 1)).reshape(3, 1)
    cur_footstep[6:9] = np.matmul(R, cur_footstep[6:9].reshape(3, 1)).reshape(3, 1)
    cur_footstep[9:12] = np.matmul(R, cur_footstep[9:12].reshape(3, 1)).reshape(3, 1)

    d1 = skew_numpy(np.vstack((cur_footstep[0, 0], cur_footstep[1, 0], cur_footstep[2, 0])))
    d2 = skew_numpy(np.vstack((cur_footstep[3, 0], cur_footstep[4, 0], cur_footstep[5, 0])))
    d3 = skew_numpy(np.vstack((cur_footstep[6, 0], cur_footstep[7, 0], cur_footstep[8, 0])))
    d4 = skew_numpy(np.vstack((cur_footstep[9, 0], cur_footstep[10, 0], cur_footstep[11, 0])))

    B[6:9, 0:3] = np.matmul(I_hat_inv, d1)
    B[6:9, 3:6] = np.matmul(I_hat_inv, d2)
    B[6:9, 6:9] = np.matmul(I_hat_inv, d3)
    B[6:9, 9:] = np.matmul(I_hat_inv, d4)

    state = np.matmul(I_identity + A * dt, cur_state) + np.matmul(B * dt, control_t) + dt * g

    return state