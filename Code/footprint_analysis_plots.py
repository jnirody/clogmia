#!/usr/bin/python3
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
from scipy.ndimage import  rotate
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)*0.55
###########################################################################
def rotate_via_numpy(x, y, radians):
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    length = min(len(x), len(y))
    m = np.dot(j, np.matrix([x[0:length-1], y[0:length-1]]))

    return (m[0], m[1])
    
def mysine(x, a1, a2, a3):
    yy = a1 * np.sin(a2 * x) + a3
    return yy
###############################################################################
condition = 'sandpaper'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
upperdir = '/home/eebrandt/projects/UChicago/fly_walking/sandpaper'
directory = upperdir + '/data/processed_data/'
datadir = upperdir + '/data/processed_data/Individual/ByStride/'
files = glob.glob(datadir + '*.csv')
byframe_dir = upperdir + '/data/processed_data/Individual/ByFrame/'
avg_file = upperdir + '/data/processed_data/fly_averages.csv'
avgdata = pd.read_csv(avg_file)
sorted_files = {}
metadata = pd.read_csv(directory + 'metadata.csv')


framerate = 240. # frames/sec
df = {}
IDdata = {}

treatments = ['glass','g150','g100','g60','g24']

for file in files:
    treatment = file.split('/')[-1].split('_')[1]
    if treatment in sorted_files.keys():
        sorted_files[treatment].append(file)
    else:
        sorted_files[treatment] = [file]
        
for treatment in sorted_files:
    print(treatment)
    df[treatment] = []
    IDdata[treatment] = []
    #meta[treatment] = []
    rel_avgdata = avgdata[avgdata['treatment']==treatment]
    #print(pd.DataFrame.from_dict(avgdata).iloc[0])
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


    rel_stance_starts = [[] for i in range(6)]
    rel_swing_starts = [[] for i in range(6)]

    avg_speed = []
    body_length = []
    avg_hl_footprint = []
    avg_ml_footprint = []
    sd_hl_footprint = []
    sd_ml_footprint = []

    
    for file in sorted_files[treatment]:
        #print(file)
        fly = file.split('/')[-1].split('_')[0]
        dataframe = pd.read_csv(file)
        #this will be the second column on the output dataframe - it associates each each observation with the specific trial
        dataframe["total_ID"] = dataframe["fly"] + "_" + dataframe["treatment"] + "_" + dataframe["iteration"].astype(str) + "_" + dataframe["trial"]
       
        
        grouped = dataframe.groupby('trial')
        for video,data in grouped:
            #print(video)
            curr_hl_footprint = []
            curr_ml_footprint = []
        
            swing_starts = [[] for i in range(6)]
            stance_starts = [[] for i in range(6)]
            PEP_locations = [[] for i in range(6)]
            AEP_locations = [[] for i in range(6)]
            swing_interval_start = []
            swing_interval_end = []
            stance_interval_start = []
            stance_interval_end = []
            
            resolution = data['limb_length'] # mm / pixel
        
            # get out all the stance and swing start times to compute out relatives
            swing_starts[5] = list(map(int,data['L1_swing_start'][data['L1_swing_start'] > 0].to_list()))
            stance_starts[5] = list(map(int,data['L1_stance_start'][data['L1_stance_start'] > 0].to_list()))
            swing_starts[4] = list(map(int,data['R1_swing_start'][data['R1_swing_start'] > 0].to_list()))
            stance_starts[4] = list(map(int,data['R1_stance_start'][data['R1_stance_start'] > 0].to_list()))
            swing_starts[3] = list(map(int,data['L2_swing_start'][data['L2_swing_start'] > 0].to_list()))
            stance_starts[3] = list(map(int,data['L2_stance_start'][data['L2_stance_start'] > 0].to_list()))
            swing_starts[2] = list(map(int,data['R2_swing_start'][data['R2_swing_start'] > 0].to_list()))
            stance_starts[2] = list(map(int,data['R2_stance_start'][data['R2_stance_start'] > 0].to_list()))
            swing_starts[1] = list(map(int,data['L3_swing_start'][data['L3_swing_start'] > 0].to_list()))
            stance_starts[1] = list(map(int,data['L3_stance_start'][data['L3_stance_start'] > 0].to_list()))
            swing_starts[0] = list(map(int,data['R3_swing_start'][data['R3_swing_start'] > 0].to_list()))
            stance_starts[0] = list(map(int,data['R3_stance_start'][data['R3_stance_start'] > 0].to_list()))
        
            leg_dict = ['R3','L3','R2','L2','R1','L1']
                
            for leg in range(4):
                for stride in range(len(stance_starts[leg])):
                    if stride > len(swing_starts[leg+2]):
                        continue
                    x = np.array(data[leg_dict[leg] + '_AEP'])[np.where(np.array(data[leg_dict[leg] + '_stance_start'] == float(stance_starts[leg][stride])))[0][0]]
                    AEP = tuple(map(float,x[1:-1].split(',')))
                    idx = np.where(np.array(swing_starts[leg+2])>stance_starts[leg][stride]-1)
                    if len(idx[0]) < 1:
                        continue
                    else:
                        idx = idx[0][0]
                        follow_swing = swing_starts[leg+2][idx]
                        x = np.array(data[leg_dict[leg+2] + '_PEP'])[np.where(np.array(data[leg_dict[leg+2] + '_swing_start'] == float(follow_swing)))[0][0]]
                        PEP = tuple(map(float,x[1:-1].split(',')))
                    if leg > 1:
                        if distance(AEP,PEP)*resolution.iloc[0] < float(avgdata['body_length'][np.where(avgdata['fly'] == fly)[0][0]][1:-1].split(',')[0])/2.:
                            curr_ml_footprint.extend([distance(AEP,PEP)*resolution])
                    else:
                        if distance(AEP,PEP)*resolution.iloc[0] < float(avgdata['body_length'][np.where(avgdata['fly'] == fly)[0][0]][1:-1].split(',')[0])/2.:
                            curr_hl_footprint.extend([distance(AEP,PEP)*resolution])
                if len(curr_hl_footprint) > 0:
                    avg_ml_footprint.append(np.mean(curr_hl_footprint))
                if len(curr_ml_footprint) > 0:
                    avg_ml_footprint.append(np.mean(curr_ml_footprint))
     
            totalID = data["total_ID"].tolist()

            idlist = [totalID[1]] * len(avg_ml_footprint)
            IDdata[treatment].extend(idlist)
            df[treatment].extend(avg_ml_footprint)
                
    #print(len(avg_ml_footprint))
    #df[treatment].extend(avg_ml_footprint)
            #ax = sns.violinplot(data=df,palette=colors,inner=None)
            #for violin,alpha in zip(ax.collections[::1], [0.5,0.5,0.5,0.5]):
                #violin.set_alpha(alpha)'
                
leaderdata = pd.DataFrame()
for treatment in treatments:
	test = pd.concat([pd.DataFrame.from_dict(IDdata[treatment]), pd.DataFrame.from_dict(df[treatment])], axis = 1)
	leaderdata = pd.concat([leaderdata, test])
leaderdata.columns = ["total_ID", "phidiff"]

             
leaderdata.to_csv('/home/eebrandt/projects/UChicago/fly_walking/sandpaper/data/leaderdata.csv', index=False) 

dd = []
for treatment in treatments:
    #print(len(df[treatment]))
    t_stat, p_val = stats.ttest_ind(df[treatment], df['glass'])
    print(mean(df[treatment]), std(df[treatment]), t_stat, p_val)
    dd.append(df[treatment])
colors = ['orange','blue','green','red','yellow']
ax = sns.violinplot(data=dd, palette=colors)
ax.set_xticklabels(treatments)
plt.savefig(upperdir+'/analysis/python_raw/followtheleader.pdf')

