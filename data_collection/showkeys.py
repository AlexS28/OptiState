import scipy.io
import matplotlib.pyplot as plt
import numpy as np

#data from the robot
#Following data might be useful, some data might be missing in some of the mat files (all zeros) so please double check before using
#enc_vel_val -> encoder velocity
#enc_val -> encoder value
#imu_accel -> imu xyz acceleration
#imu_ang_rate -> imu angular rate estimation
#time -> time stamp
estdata = scipy.io.loadmat("Estimated_traj23.mat") 
print(estdata.keys())

#data from vicon (groud truth)
#Note: The mat file where the index matches the Estimated_traj mat file are paired, however the starting time of the trajectory are not alligned with Estimated_traj mat file, calibration is needed.
#object_pos -> pose of the robot
#mk_pos -> xyz position of the markers on the robot
#time -> time stamp
vicondata = scipy.io.loadmat("vicon_data23.mat")
print(vicondata.keys())
