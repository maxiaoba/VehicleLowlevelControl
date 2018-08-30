import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

log_dir = "/Users/xiaobaima/Dropbox/SISL/RARL/JustEgoGrey/TestBed/Data/Normal/"
names = []
# for policy in os.listdir(logdir+"Disturbance"):
# 	if not (policy == ".DS_Store"):
# 		names.append(policy[:end-4])
names = ["fsp_1e3"]

figureid = 1
for name in names:
	fig = plt.figure(figureid, figsize=(8, 16))

	daccs = np.loadtxt(log_dir+"Disturbance/"+name+"_daccs"+".txt")
	dsteers = np.loadtxt(log_dir+"Disturbance/"+name+"_dsteers"+".txt")

	num_bins = 40

	plt.subplot(1,2,1)
	n, bins, patches = plt.hist(daccs, num_bins, facecolor='blue', alpha=0.5,range=(-1.2,1.2))
	plt.title(name+'_br')
	plt.xlabel('dacc [m/s^2]')
	plt.subplot(1,2,2)
	n, bins, patches = plt.hist(dsteers, num_bins, facecolor='blue', alpha=0.5,range=(-1.2,1.2))
	plt.xlabel('dsteer [rad]')

	fig.savefig(logdir+"Plot/Disturbance/"+name+'.png')
	plt.close(fig)