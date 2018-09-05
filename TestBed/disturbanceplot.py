import tensorflow as tf
import joblib
from matplotlib import pyplot as plt
import numpy as np

log_dir = "/Users/xiaobaima/Dropbox/SISL/RARL/JustEgoGrey/TestBed/Data/Sep1/Ad05RTheta1RSteer05Abs0FR200/"
names = []
names = ["fsp_1e3"]
for name in names:
	states = np.loadtxt(log_dir+"Disturbance/"+name+"_reg_states"+".txt")
	daccs = np.loadtxt(log_dir+"Disturbance/"+name+"_reg_daccs"+".txt")
	dsteers = np.loadtxt(log_dir+"Disturbance/"+name+"_reg_dsteers"+".txt")

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

	fig.savefig(log_dir+"Plot/Disturbance/"+name+"_reg_scatter_y.pdf")
	plt.close(fig)
