import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def plot3(n,x,y,z):

    fig, ax = plt.subplots()

    line0, = ax.plot(range(n),x,label="x")
    line1, = ax.plot(range(n),y,label="y")
    line2, = ax.plot(range(n),z,label="z")


    ax.legend()
    plt.show()
    
def plot4(n,x,y,z,w):

    fig, ax = plt.subplots()

    line0, = ax.plot(range(n),x,label="0")
    line1, = ax.plot(range(n),y,label="1")
    line2, = ax.plot(range(n),z,label="2")
    line2, = ax.plot(range(n),w,label="3")


    ax.legend()
    plt.show()
    
def plot1(n,x):
    fig, ax = plt.subplots()

    line0, = ax.plot(range(n),x)
    plt.show()



estdata = scipy.io.loadmat("Estimated_traj23.mat")
print(estdata["est_val"].shape)

print(estdata['time'])

print(estdata["contact"])
cut = 400
#plot4(cut, estdata["contact"][:cut,0], estdata["contact"][:cut,1],estdata["contact"][:cut,2],estdata["contact"][:cut,3])
plot1(cut, estdata["contact"][:cut,0])
cut = 800
plot3(cut, estdata["est_val"][:cut,0], estdata["est_val"][:cut,1], estdata["est_val"][:cut,2])
plot3(cut, estdata["camera"][:cut,3], estdata["camera"][:cut,4], estdata["camera"][:cut,5])



print(estdata['imu_accel'])
plot1(cut, estdata["imu_accel"][:cut,2])
plot1(cut, estdata["wrench"][:cut,2])
plot1(cut, estdata["filtered_force"][:cut,0])

"""
print(estdata['enc_val'])
for i in range(12):
    plot1(cut, estdata["enc_val"][:cut,i])
"""
"""
fig, ax = plt.subplots()

line0, = ax.plot(range(lsedata.shape[0]),lsedata[:,0],label="x")
line1, = ax.plot(range(lsedata.shape[0]),lsedata[:,1],label="y")
line2, = ax.plot(range(lsedata.shape[0]),lsedata[:,2],label="z")


ax.legend()

fig1, ax1 = plt.subplots()

line0, = ax1.plot(range(lsedata.shape[0]),CMdata[:,0],label="x")
line1, = ax1.plot(range(lsedata.shape[0]),CMdata[:,1],label="y")
line2, = ax1.plot(range(lsedata.shape[0]),CMdata[:,2],label="z")


ax1.legend()
plt.show()
"""


mocapdata = scipy.io.loadmat("vicon_data23.mat")
print(mocapdata['object_pos'].shape)
cut = 1700
plot3(cut, mocapdata['object_pos'][:cut,0], mocapdata['object_pos'][:cut,1], mocapdata['object_pos'][:cut,2])









