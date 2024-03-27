###########################################################################
# Makes averaged gait diagram.
# This makes Fig1c, Fig S3
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
from scipy.stats import mode, circmean, circstd, circvar
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.gridspec as gridspec
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
def gauss(x,mu,sigma,A,c):
    return A*exp(-(x-mu)**2/2/sigma**2) + c
###########################################################################
def von_mises_distribution(x, mu, kappa):
    return stats.vonmises.pdf(x, kappa, mu)
###########################################################################

condition = 'sandpaper'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/ForAnalysis/' + condition + '/Individual/ByStride/'
files = glob.glob(datadir + '*.csv')
sorted_files = {}

for file in files:
    treatment = file.split('/')[-1].split('_')[1]
    if treatment in sorted_files.keys():
        sorted_files[treatment].append(file)
    else:
        sorted_files[treatment] = [file]
avg_file = upperdir + '/Data/ForAnalysis/' + condition + '/fly_averages.csv'
avgdata = pd.read_csv(avg_file)

for treatment in sorted_files:
    print(treatment)

    rel_avgdata = avgdata[avgdata['treatment']==treatment]
    rel_swing_lengths = [[] for i in range(6)]
    rel_stance_lengths = [[] for i in range(6)]

    rel_swing_lengths[0] = [np.mean((rel_avgdata['L1_swing']/rel_avgdata['L1_period']).to_list()),np.std((rel_avgdata['L1_swing']/rel_avgdata['L1_period']).to_list())]
    rel_stance_lengths[0] = [np.mean((rel_avgdata['L1_stance']/rel_avgdata['L1_period']).to_list()),np.std((rel_avgdata['L1_stance']/rel_avgdata['L1_period']).to_list())]
    rel_swing_lengths[3] = [np.mean((rel_avgdata['R1_swing']/rel_avgdata['R1_period']).to_list()),np.std((rel_avgdata['R1_swing']/rel_avgdata['R1_period']).to_list())]
    rel_stance_lengths[3] = [np.mean((rel_avgdata['R1_stance']/rel_avgdata['R1_period']).to_list()),np.std((rel_avgdata['R1_stance']/rel_avgdata['R1_period']).to_list())]
    rel_swing_lengths[1] = [np.mean((rel_avgdata['L2_swing']/rel_avgdata['L2_period']).to_list()),np.std((rel_avgdata['L2_swing']/rel_avgdata['L2_period']).to_list())]
    rel_stance_lengths[1] = [np.mean((rel_avgdata['L2_stance']/rel_avgdata['L2_period']).to_list()),np.std((rel_avgdata['L2_stance']/rel_avgdata['L2_period']).to_list())]
    rel_swing_lengths[4] = [np.mean((rel_avgdata['R2_swing']/rel_avgdata['R2_period']).to_list()),np.std((rel_avgdata['R2_swing']/rel_avgdata['R2_period']).to_list())]
    rel_stance_lengths[4] = [np.mean((rel_avgdata['R2_stance']/rel_avgdata['R2_period']).to_list()),np.std((rel_avgdata['R2_stance']/rel_avgdata['R2_period']).to_list())]
    rel_swing_lengths[2] = [np.mean((rel_avgdata['L3_swing']/rel_avgdata['L3_period']).to_list()),np.std((rel_avgdata['L3_swing']/rel_avgdata['L3_period']).to_list())]
    rel_stance_lengths[2] = [np.mean((rel_avgdata['L3_stance']/rel_avgdata['L3_period']).to_list()),np.std((rel_avgdata['L3_stance']/rel_avgdata['L3_period']).to_list())]
    rel_swing_lengths[5] = [np.mean((rel_avgdata['R3_swing']/rel_avgdata['R3_period']).to_list()),np.std((rel_avgdata['R3_swing']/rel_avgdata['R3_period']).to_list())]
    rel_stance_lengths[5] = [np.mean((rel_avgdata['R3_stance']/rel_avgdata['R3_period']).to_list()),np.std((rel_avgdata['R3_stance']/rel_avgdata['R3_period']).to_list())]

    rel_swing_starts = [[] for i in range(6)]
    rel_stance_starts = [[] for i in range(6)]
    
    for file in sorted_files[treatment]:
        fly = file.split('/')[-1].split('_')[0]
        dataframe = pd.read_csv(file)
        
        grouped = dataframe.groupby('trial')
        for video,data in grouped:
            swing_starts = [[] for i in range(6)]
            stance_starts = [[] for i in range(6)]
            interval_start = []
            interval_end = []
            swing_starts[0] = data['L1_swing_start'][data['L1_swing_start'] > 0].to_list()
            stance_starts[0] = data['L1_stance_start'][data['L1_stance_start'] > 0].to_list()
            # get out all the stance and swing start times to compute out relatives
            swing_starts[3] = data['R1_swing_start'][data['R1_swing_start'] > 0].to_list()
            stance_starts[3] = data['R1_stance_start'][data['R1_stance_start'] > 0].to_list()
            swing_starts[1] = data['L2_swing_start'][data['L2_swing_start'] > 0].to_list()
            stance_starts[1] = data['L2_stance_start'][data['L2_stance_start'] > 0].to_list()
            swing_starts[4] = data['R2_swing_start'][data['R2_swing_start'] > 0].to_list()
            stance_starts[4] = data['R2_stance_start'][data['R2_stance_start'] > 0].to_list()
            swing_starts[2] = data['L3_swing_start'][data['L3_swing_start'] > 0].to_list()
            stance_starts[2] = data['L3_stance_start'][data['L3_stance_start'] > 0].to_list()
            swing_starts[5] = data['R3_swing_start'][data['R3_swing_start'] > 0].to_list()
            stance_starts[5] = data['R3_stance_start'][data['R3_stance_start'] > 0].to_list()
            
            if len(swing_starts[0]) < 2:
                continue
            interval_start.append(swing_starts[0][0])
            interval_end.append(swing_starts[0][1])
            for i in range(1,len(swing_starts[0])-1):
                interval_start.append(swing_starts[0][i])
                interval_end.append(swing_starts[0][i+1])

            for i in range(len(swing_starts)): #leg
                for j in range(len(swing_starts[i])): #stride
                    k = np.where(np.array(interval_start) < swing_starts[i][j]+0.1)[0]
                    if len(k) == 0:
                        continue
                    k = max(k)
                    if swing_starts[i][j] > interval_end[k]:
                        continue
                    curr_interval = [interval_start[k],interval_end[k]]
                    rel_swing_starts[i].append((float(swing_starts[i][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
                    
            for i in range(len(stance_starts)): #leg
                for j in range(len(stance_starts[i])): #stride
                    k = np.where(np.array(interval_end) > stance_starts[i][j]+0.1)[0]
                    if len(k) == 0:
                        continue
                    k = max(k)
                    if stance_starts[i][j] < interval_start[k]:
                            continue
                    curr_interval = [interval_start[k],interval_end[k]]
                    rel_stance_starts[i].append((float(stance_starts[i][j]-curr_interval[0]))/ (curr_interval[1]-curr_interval[0]))
                    
    lines = [[] for i in range(6)]
    sdshade = [[] for i in range(6)]
    for i in range(len(rel_swing_starts)):
        mode = stats.mode(rel_swing_starts[i])
        m = mode[0]
        if m == 0.0:
            c = -1
        elif m == 1.0:
            c = 1
        else:
            c = 0
        expected=(m,0.1,5,c)
        y,x,_ = hist(rel_swing_starts[i],50)
        x = (x[1:]+x[:-1])/2
        params1,cov=curve_fit(gauss,x,y,expected,maxfev=100000)
        if abs(params1[0]) > 1:
            params1[0] = circmean(rel_swing_starts[i], high=1, low=0)
            params1[1] = circstd(rel_swing_starts[i], high=1, low=0)
        sigma=sqrt(diag(cov))
        
        if i == 0:
            params1[0] = 0
            params1[1] = 0
        sdshade[i] = [[max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]*2-rel_swing_lengths[i][1]]),max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]*2+rel_swing_lengths[i][1]])],[max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]-params1[1]]),max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]+params1[1]])],[max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][1]]),max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-rel_stance_lengths[i][0]+rel_swing_lengths[i][1]])],[max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-params1[1]]),max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]+params1[1]])], [params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][1],params1[0]-rel_stance_lengths[i][0]+rel_swing_lengths[i][1]],[params1[0]-params1[1],params1[0]+params1[1]], [min([1.5,params1[0]+rel_swing_lengths[i][0]-rel_swing_lengths[i][1]]),min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_swing_lengths[i][1]])],[min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]-params1[1]]),min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+params1[1]])],[min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]-rel_swing_lengths[i][1]]),min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]+rel_swing_lengths[i][1]])]]
        lines[i] = [[max([-0.5,params1[0] -rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]*2]),max([-0.5,params1[0]-rel_stance_lengths[i][0]-rel_swing_lengths[i][0]*2-rel_stance_lengths[i][0]])],[max([-0.5,params1[0] -rel_stance_lengths[i][0]-rel_swing_lengths[i][0]-rel_stance_lengths[i][0]]), max([-0.5,params1[0] -rel_stance_lengths[i][0]-rel_swing_lengths[i][0]])],[params1[0]-rel_stance_lengths[i][0],params1[0]], [min([1.5,params1[0]+rel_swing_lengths[i][0]]),min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]])],[min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]]),min([1.5,params1[0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]+rel_swing_lengths[i][0]+rel_stance_lengths[i][0]])]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axvspan(0,1, facecolor='grey', alpha=0.4)
    for leg in range(len(lines)):
        for k in range(len(lines[leg])):
            if lines[leg][k][1] - lines[leg][k][0] == 0:
                continue
            plt.plot(lines[leg][k],[leg*-1,leg*-1],'k',linewidth=10, zorder=1)
        for m in range(len(sdshade[leg])):
            if sdshade[leg][m][1] - sdshade[leg][m][0] == 0:
                continue
            plt.plot(sdshade[leg][m],[leg*-1,leg*-1],'r',linewidth=3,alpha=0.8,zorder=1)
    ax.set_yticklabels(['','R3','R2','R1','L3','L2','L1'])
    ax.set_xticks([-0.5,0,0.5,1,1.5])
    ax.set_xlim([-0.5,1.5])
    plt.axvspan(-0.54,-0.5, facecolor='white', zorder=2)
    plt.axvspan(1.5,1.54, facecolor='white', zorder=2)
    plt.yticks(fontname='Georgia',fontsize=18)
    plt.xticks(fontname='Georgia',fontsize=18)
    ax.set_xlabel('Gait cycle',fontname='Georgia', fontsize=24)
    plt.tight_layout()
    plt.savefig(upperdir+'/Figures/avggaitdiagram_' + condition + '_' + treatment + '.pdf')
