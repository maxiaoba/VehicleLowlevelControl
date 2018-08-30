import matplotlib.pyplot as plt
import numpy as np
import os

names = []
for policy in os.listdir("./Policy"):
	print(policy != '.DS_Store')
	if policy != '.DS_Store':
		names.append(policy[:-4:])

figureid = 1
data = []
for name in names:
	fig = plt.figure(figureid, figsize=(8, 6))

	axes = plt.gca()
	axes.set_xlim([-1,1])
	axes.set_ylim([-200,800])
	data_i = np.loadtxt('Data/'+name+'_rewards_d_pareto10'+'.txt')
	data.append(data_i)

plt.boxplot(data,labels=names)
plt.title(name)
plt.ylabel('single path undiscounted reward')
figureid += 1
fig.savefig('Plot/'+'Pareto10'+'.png')
plt.close(fig)