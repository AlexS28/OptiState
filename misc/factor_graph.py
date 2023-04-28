import gtsam
import numpy as np
from functools import partial
from typing import List, Optional
from settings import INITIAL_PARAMS
import copy
# This class will fuse multiple sources to create a single estimate of the measurements
# We will estimate z, dthz, vx, and vy. Each of these parameters are estimated from two sources.

class StateFusion:
    def __init__(self):
        # [z, dthz, vx, vy]
        self.gtsam_keys = [gtsam.symbol('x', k) for k in range(4)]
        # initialize value container
        # New Values container
        self.values = gtsam.Values()
        # Add initial estimates to the Values container
        for i in range(4):
            self.values.insert(self.gtsam_keys[i], np.array([0.0]))
        # initialize factor graph
        self.factor_graph = gtsam.NonlinearFactorGraph()
        # initialize noise models, we first get the measurement noise from the initialized covariance
        # R noise: thx^imu, thy^imu, thz^imu, z^odom, z^lidar, dthx^imu, dthy^imu, dthz^imu, dthz^flow, vx^flow, vy^flow, vx^odom, vy^odom, vz^odom
        # [[z^odom, z^lidar,dthz^imu, dthz^flow,vx^flow, vx^odom,vy^flow,vy^odom]
        measurement_noise = INITIAL_PARAMS.R_FACTOR_GRAPH
        self.noise_models = []
        for i in range(8):
            meas_noise = measurement_noise[i]
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([meas_noise]))
            self.noise_models.append(noise_model)

        # starting measurements
        self.measurement = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

        # fused measurements and covariance
        self.measurement_fused = [0.0,0.0,0.0,0.0]
        self.measurement_fused_cov = [0.0,0.0,0.0,0.0]

        # initialize Factors in factor graph
        for i in range(4):
            noise_model1 = self.noise_models[i*2]
            noise_model2 = self.noise_models[i*2+1]
            measurement1 = self.measurement[i*2]
            measurement2 = self.measurement[i*2+1]
            gf1 = gtsam.CustomFactor(noise_model1, [self.gtsam_keys[i]], partial(self.error_func, np.array([measurement1])))
            self.factor_graph.add(gf1)
            gf2 = gtsam.CustomFactor(noise_model2, [self.gtsam_keys[i]], partial(self.error_func, np.array([measurement2])))
            self.factor_graph.add(gf2)

    def update_fused_measurement(self):
        # z is the measurement after fusion of multiple sources, it's a 4x1 array [z, dthz, vx, vy]
        for i in range(4):
            self.values.update(self.gtsam_keys[i], self.measurement_fused[i])

    def update_raw_measurement(self, measurement):
        # measurement should me [z ^ odom, z ^ lidar, dthz ^ imu, dthz ^ flow, vx ^ flow, vx ^ odom, vy ^ flow, vy ^ odom]
        self.measurement = measurement

    def update_measurement_noise(self, R_factor_graph):
        # R_factor_graph, which is the covariance noise of the invidiual sensor measurements should be
        # [z^odom, z^lidar, dthz^imu, dthz^flow, vx^flow, vx^odom, vy^flow, vy^odom]
        for i in range(4):
            meas_noise = R_factor_graph[i]
            noise_models = copy.deepcopy(self.noise_models[i])
            noise_models[0] = gtsam.noiseModel.Diagonal.Sigmas(np.array([meas_noise[0]]))
            noise_models[1] = gtsam.noiseModel.Diagonal.Sigmas(np.array([meas_noise[1]]))
            self.noise_models[i] = noise_models

    def update_factors(self):
        # call this function AFTER update_raw_measurement, and update_measurement_noise
        for i in range(4):
            noise_model1 = self.noise_models[i*2]
            noise_model2 = self.noise_models[i*2+1]
            measurement1 = self.measurement[i*2]
            measurement2 = self.measurement[i*2+1]
            gf1 = gtsam.CustomFactor(noise_model1, [self.gtsam_keys[i]], partial(self.error_func, np.array([measurement1])))
            gf2 = gtsam.CustomFactor(noise_model2, [self.gtsam_keys[i]], partial(self.error_func, np.array([measurement2])))
            self.factor_graph.replace(i*2,gf1)
            self.factor_graph.replace(i*2+1,gf2)

    def update_factor_graph(self, measurement, R_factor_graph):
        self.update_raw_measurement(measurement)
        self.update_measurement_noise(R_factor_graph)
        self.update_factors()

    def solve_factor_graph(self, measurement, R_factor_graph):
        # measurement should me [z ^ odom, z ^ lidar, dthz ^ imu, dthz ^ flow, vx ^ flow, vx ^ odom, vy ^ flow, vy ^ odom]
        # R_factor_graph should be [z^odom, z^lidar, dthz^imu, dthz^flow, vx^flow, vx^odom, vy^flow, vy^odom]
        self.update_factor_graph(measurement, R_factor_graph)

        # solve the factor graph
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.factor_graph, self.values)
        self.result = optimizer.optimize()
        marginals = gtsam.Marginals(self.factor_graph, self.result)

        # update the fused measurements, and the values for the factor graph
        for i in range(4):
            self.measurement_fused[i] = self.result.atVector(self.gtsam_keys[i])[0]
            self.measurement_fused_cov[i] = marginals.marginalCovariance(self.gtsam_keys[i])[0][0]
        self.update_fused_measurement()

        return self.measurement_fused, self.measurement_fused_cov

    def error_func(self, measurement: np.ndarray, this: gtsam.CustomFactor,
                  values: gtsam.Values,
                  jacobians: Optional[List[np.ndarray]]) -> float:
        """GPS Factor error function
        :param measurement: GPS measurement, to be filled with `partial`
        :param this: gtsam.CustomFactor handle
        :param values: gtsam.Values
        :param jacobians: Optional list of Jacobians
        :return: the unwhitened error
        """
        key = this.keys()[0]
        estimate = values.atVector(key)
        error = estimate - measurement
        if jacobians is not None:
            jacobians[0] = 1.0

        return error