#!/usr/bin/python3
# Plot of walking speed distribution between different conditions
# Makes Figure S4a
###########################################################################

import re, math, sys, os, random
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob, csv
from scipy.stats import mode
import seaborn as sns
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
def gauss(x,mu,sigma,A,c):
    return A*exp(-(x-mu)**2/2/sigma**2) + c
###########################################################################
condition = 'sandpaper'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/data/processed_data/' + 'Individual/ByFrame/'
files = glob.glob(datadir + '*.csv')
avg_file = upperdir + '/data/processed_data/' + '/fly_averages.csv'
avgdata = pd.read_csv(avg_file)
sorted_files = {}

treatments = ['glass','g150','g100','g60','g24']

for file in files:
	treatment = file.split('/')[-1].split('_')[1]
	if treatment in sorted_files.keys():
		sorted_files[treatment].append(file)
	else:
		sorted_files[treatment] = [file]

COM_speeds = {}
combined = pd.DataFrame()
speeddata = pd.DataFrame()
#nums = pd.DataFrame()

for treatment in treatments:
	print(treatment)
	for file in sorted_files[treatment]:
		#print(file)
		fly = file.split('/')[-1].split('_')[0]
		dataframe = pd.read_csv(file)
		dataframe['total_ID'] = dataframe['fly'].astype(str) + '_' + dataframe['treatment'].astype(str) + '_' + dataframe['iteration'].astype(str) + '_' + dataframe['trial'].astype(str)
		#print(dataframe['trial'])
		grouped = dataframe.groupby('trial')
		
		#print(dataframe)
		#vidcount = 1
		for video,data in grouped:
			#print(video)
			temp_speeds = data['COM_speed']
			temp_fulltrial = data['total_ID']
			nums = list(range(len(temp_speeds)))
			combined = pd.concat([temp_fulltrial, temp_speeds], axis = 1)
			combined['speed_count'] = np.arange(combined.shape[0])
			#print(combined)
			combined = combined[combined['COM_speed'] >0]
			#print(combined['total_ID'].iloc[1])
			#print(combined[treatment])
			#combined = pd.DataFrame(temp_speeds, temp_fulltrial, axis = 1
			#if treatment in COM_speeds.keys():
			#	COM_speeds[treatment].extend([float(x) for x in temp_speeds if float(x) >0])
			#else:
		     #COM_speeds[treatment] == [float(x) for x in temp_speeds if float(x) > 0]
			COM_speeds[treatment] = [float(x) for x in temp_speeds if float(x) > 0]
			

			speeddata = pd.concat([speeddata, combined])
		
#print(len(COM_speeds['glass']))
#print(speeddata)

#for treatment in treatments:
#	speeddata = pd.concat(combined)
	
#speeddata = speeddata.drop(speeddata.columns[[0]], axis=1)
	
speeddata.to_csv('/home/eebrandt/projects/UChicago/fly_walking/sandpaper/data/speed.csv', index=False) 

for treatment in treatments:
	t_stat, p_val = stats.ttest_ind(COM_speeds[treatment], COM_speeds['glass'])
	#print(mean(COM_speeds[treatment]), std(COM_speeds[treatment]), t_stat, p_val, treatment)
	
	

#fig,axes = plt.subplots()
#colors = ['orange','blue','green','red','yellow']
#sns.swarmplot(data=COM_speeds, palette=colors)
#plt.ylabel('Walking speed (bl/s)',fontsize=19, fontname='Georgia')
#axes.set_xticklabels(treatments, fontsize=14, fontname='Georgia')
#plt.savefig('/home/eebrandt/projects/UChicago/fly_walking/sandpaper/analysis/walkingspeed_across_substrates.pdf')
            
            
