# This file evaluates the Kalman filter
import pickle
import matplotlib.pyplot as plt
import numpy as np
from settings import INITIAL_PARAMS
import os
from kalman_filter.kalman_filter import Kalman_Filter

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
traj_num = 0
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

KF = Kalman_Filter()



