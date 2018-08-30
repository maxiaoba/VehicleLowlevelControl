import tensorflow as tf
import joblib
from matplotlib import pyplot as plt
import numpy as np

from os import sys, path
sys.path.append(path.abspath(path.join(path.dirname(__file__), '../Fsp')))
sys.path.append(path.abspath(path.join(path.dirname(__file__), '../CommonFiles')))

log_dir = "/Users/xiaobaima/Dropbox/SISL/RARL/JustEgoGrey/TestBed/Data/Normal/"
policy = "fsp_1e3"
with tf.Session() as sess:
	data = joblib.load(log_dir+"Policy/"+policy+".pkl")
	buffer1 = data["buffer1"]
	buffer2 = data["buffer2"]

	states,actions = buffer1.get_data()
	y = [state[0] for state in states]
	v =	[state[1] for state in states]
	theta = [state[2] for state in states]
	k = [state[3] for state in states]
	acc = [action[0] for  action in actions]
	steer = [action[1] for action in actions]

	states2,actions2 = buffer2.get_data()
	y2 = [state[0] for state in states2]
	dacc = [action[0] for  action in actions2]
	dsteer = [action[1] for action in actions2]


	fig = plt.figure()
	plt.scatter(v,y)
	plt.xlabel('v')
	plt.ylabel('y')
	fig.savefig(log_dir+"Plot/Buffer/"+policy+"/v_y.pdf")
	plt.close(fig)

	fig = plt.figure()
	plt.scatter(theta,y)
	plt.xlabel('theta')
	plt.ylabel('y')
	fig.savefig(log_dir+"Plot/Buffer/"+policy+"/theta_y.pdf")
	plt.close(fig)

	fig = plt.figure()
	plt.scatter(k,y)
	plt.xlabel('k')
	plt.ylabel('y')
	fig.savefig(log_dir+"Plot/Buffer/"+policy+"/k_y.pdf")
	plt.close(fig)

	fig = plt.figure()
	plt.scatter(acc,y)
	plt.xlabel('acc')
	plt.ylabel('y')
	fig.savefig(log_dir+"Plot/Buffer/"+policy+"/acc_y.pdf")
	plt.close(fig)

	fig = plt.figure()
	plt.scatter(steer,y)
	plt.xlabel('steer')
	plt.ylabel('y')
	fig.savefig(log_dir+"Plot/Buffer/"+policy+"/steer_y.pdf")
	plt.close(fig)

	fig = plt.figure()
	plt.scatter(dacc,y2)
	plt.xlabel('dacc')
	plt.ylabel('y')
	fig.savefig(log_dir+"Plot/Buffer/"+policy+"/dacc_y.pdf")
	plt.close(fig)

	fig = plt.figure()
	plt.scatter(dsteer,y2)
	plt.xlabel('dsteer')
	plt.ylabel('y')
	fig.savefig(log_dir+"Plot/Buffer/"+policy+"/dsteer_y.pdf")
	plt.close(fig)
