# Plot of walking speed distribution between different conditions
# Makes Figure S4a
###########################################################################
#!/usr/bin/python
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
datadir = upperdir + '/Data/ForAnalysis/' + condition + '/Individual/ByFrame/'
files = glob.glob(datadir + '*.csv')
avg_file = upperdir + '/Data/ForAnalysis/' + condition + '/fly_averages.csv'
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

for treatment in treatments:
    print(treatment)
    for file in sorted_files[treatment]:
        fly = file.split('/')[-1].split('_')[0]
        dataframe = pd.read_csv(file)
        grouped = dataframe.groupby('trial')

        for video,data in grouped:
            temp_speeds = data['COM_speed']
            COM_speeds[treatment] = [float(x) for x in temp_speeds if float(x) > 0]

for treatment in treatments:
    t_stat, p_val = stats.ttest_ind(COM_speeds[treatment], COM_speeds['glass'])
    print(mean(COM_speeds[treatment]), std(COM_speeds[treatment]), t_stat, p_val, treatment)

fig,axes = plt.subplots()
colors = ['orange','blue','green','red','yellow']
sns.swarmplot(data=COM_speeds, palette=colors)
plt.ylabel('Walking speed (bl/s)',fontsize=19, fontname='Georgia')
axes.set_xticklabels(treatments, fontsize=14, fontname='Georgia')
plt.savefig(upperdir + '/Figures/walkingspeed_across_substrates.pdf')
            
            
