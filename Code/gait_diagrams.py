# Just for exposition, not in paper.
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
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
condition = 'ablatement'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
datadir = upperdir + '/Data/ForAnalysis/' + condition + '/Individual/ByFrame/'
files = glob.glob(datadir + '*.csv')

sides = ['L','R']

colors = ['m','c','y','r','b','g','k']

for file in files:
    print(file)
    s = '-'
    fly = s.join(file.split('/')[-1].split('_')[0:-1])
    print(fly)
    dataframe = pd.read_csv(file, index_col=False)
    grouped = dataframe.groupby('trial')
    for trial,data in grouped:
        print(trial)
        swing_starts = [[]]*6
        stance_starts = [[]]*6
        lines = {}
        fig1, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]})
        for leg_pair in range(3):
            for side in range(2):
                lines[2*(leg_pair)+side] = []
                leg = sides[side] + str(leg_pair+1)
                leg_down = data[leg + '_leg_down'].to_list()
                swing_starts[2*(leg_pair)+side] = (np.where(np.diff(leg_down)==-1)[0]+1).tolist()
                stance_starts[2*(leg_pair)+side] = (np.where(np.diff(leg_down)==1)[0]+2).tolist()
                num_legs_down = data['num_feet_down'].to_list()
                if stance_starts[2*(leg_pair)+side][0] > swing_starts[2*(leg_pair)+side][0]:
                    holder = [0] + stance_starts[2*(leg_pair)+side]
                    stance_starts[2*(leg_pair)+side] = holder
                num_swings = len(swing_starts[2*(leg_pair)+side])
                num_stances = len(stance_starts[2*(leg_pair)+side])
                z = [num_swings,num_stances]
                for j in range(min(z)):
                    lines[2*(leg_pair)+side].append([swing_starts[2*(leg_pair)+side][j],stance_starts[2*(leg_pair)+side][j]])
        for leg in range(6):
            for k in range(len(lines[leg])):
                ax1.plot(lines[leg][k],[leg,leg],'k',linewidth=4)
            ax1.set_ylim(-1,6)
            ax1.set_yticks(np.arange(0,6,1))
            ax1.set_yticks(np.arange(0,6,1), ['L1','R1','L2','R2','L3','R3'])
        for i in range(len(num_legs_down)-1):
            ax2.axvline(i,color=colors[int(num_legs_down[i])],linewidth=4)
            ax2.set_yticks([])
        plt.savefig(upperdir + '/Figures/GaitDiagrams/' + condition + '_' + fly + '-' + str(trial) + '.pdf')
