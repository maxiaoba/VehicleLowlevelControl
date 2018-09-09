import tensorflow as tf
import joblib
from matplotlib import pyplot as plt
import numpy as np

log_dir = "/Users/xiaobaima/Dropbox/SISL/RARL/JustEgoGrey/TestBed/Data/Sep1/Ad05RTheta1RSteer05Abs0FR200/"
names = []
names = ["fsp_2e3","fsp_3e3","fsp_4e3"]
reg = True
if reg:
	for (i,name) in enumerate(names):
		names[i] = name+"_reg"

for name in names:
	states = np.loadtxt(log_dir+"Disturbance/"+name+"_states"+".txt")
	daccs = np.loadtxt(log_dir+"Disturbance/"+name+"_daccs"+".txt")
	dsteers = np.loadtxt(log_dir+"Disturbance/"+name+"_dsteers"+".txt")

	ys = [state[0] for state in states]

	fig = plt.figure()
	plt.subplot(1,2,1)
	plt.scatter(daccs,ys)
	plt.xlabel('dacc')
	plt.ylabel('y')

	plt.subplot(1,2,2)
	plt.scatter(dsteers,ys)
	plt.xlabel('dsteer')
	plt.ylabel('y')

	fig.savefig(log_dir+"Plot/Disturbance/"+name+"_scatter_y.pdf")
	plt.close(fig)
