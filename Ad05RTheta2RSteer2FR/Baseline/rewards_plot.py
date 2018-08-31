import tensorflow as tf
import joblib
from matplotlib import pyplot as plt
import numpy as np
import os
import csv

from statsmodels.nonparametric.smoothers_lowess import lowess

log_dir = "./col50tanh/progress"

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 

fig = plt.figure(1,figsize=(6,6))
lines = []

reader = csv.DictReader(open(log_dir+'.csv'))
AvgRewards = []
MinRewards = []
MaxRewards = []
AvgDisRewards = []
for row in reader:
	# print(row)
	AvgRewards.append(row['AverageReturn'])
	MinRewards.append(row['MinReturn'])
	MaxRewards.append(row['MaxReturn'])
	AvgDisRewards.append(row['AverageDiscountedReturn'])
x = range(len(AvgRewards))

result = lowess(AvgRewards,x)
AvgRewards_smooth = result[:,1]

line, = plt.plot(x, AvgRewards_smooth)
lines.append(line)
# plt.legend(handles=lines)
plt.xlabel('Iteration')
plt.ylabel('Average Undiscounted Path Reward')
# plt.show()
fig.savefig(log_dir+'.pdf')
plt.close(fig)



