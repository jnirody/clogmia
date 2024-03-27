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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
import scipy
from scipy.stats import linregress
from scipy.optimize import curve_fit

# This makes all the plots shown in Figure 2, S1, S2
# This makes the plot in Figure 3 (not inset)

####################################################
def func(x, a, b, c):
    return a*x**(b)+c
####################################################
condition = 'sandpaper'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
files = glob.glob(upperdir + '/Data/ForAnalysis/' + condition + '/' + 'compiled*.csv')
alldata = {}

for file in files:
    treatment = file.split('/')[-1].split('_')[-1][:-4]
    alldata[treatment] = pd.read_csv(file)

shapes = ['.', '+', 'X', 'D', 'v']
colors = ['r','g','b','y','k']
# Look at how each leg pair correlates with speed
count = -1
fig=plt.figure(figsize=(10,7))
for treatment in alldata:
    count += 1
    first_leg_speed = list(map(float,alldata[treatment]['R1_stride_speed'].tolist()))
    first_leg_speed.extend(list(map(float,alldata[treatment]['L1_stride_speed'].tolist())))
    second_leg_speed = list(map(float,alldata[treatment]['R2_stride_speed'].tolist()))
    second_leg_speed.extend(list(map(float,alldata[treatment]['L2_stride_speed'].tolist())))
    third_leg_speed = list(map(float,alldata[treatment]['R3_stride_speed'].tolist()))
    third_leg_speed.extend(list(map(float,alldata[treatment]['L3_stride_speed'].tolist())))

    loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

    first_step = list(map(float,alldata[treatment]['R1_step_length'].tolist()))
    first_step.extend(list(map(float,alldata[treatment]['L1_step_length'].tolist())))
    idx1 = (np.array(first_step) > 0) & (np.array(first_leg_speed) > 2)

    second_step = list(map(float,alldata[treatment]['R2_step_length'].tolist()))
    second_step.extend(list(map(float,alldata[treatment]['L2_step_length'].tolist())))
    idx2 = (np.array(second_step) > 0) & (np.array(second_leg_speed) > 2)

    third_step = list(map(float,alldata[treatment]['R3_step_length'].tolist()))
    third_step.extend(list(map(float,alldata[treatment]['L3_step_length'].tolist())))
    idx3 = (np.array(third_step) > 0) & (np.array(third_leg_speed) > 2)

    loc_legs_step = first_step + second_step + third_step
    idx = (np.array(loc_legs_step) > 0) & (np.array(loc_legs_speed) > 2)


    xdata = np.array(first_leg_speed)[idx1]
    ydata = np.array(first_step)[idx1]
    pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(xdata,ydata, color = 'blue', linestyle = '', marker = shape, alpha=0.4,label='First leg pair, ' + treatment + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')

    xdata = np.array(second_leg_speed)[idx2]
    ydata = np.array(second_step)[idx2]
    pars, cov = curve_fit(func,xdata,ydata, p0=[1, -0.1, 1], maxfev=100000)
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(xdata,ydata, color = 'green', linestyle = '', marker = shape, alpha=0.4, label='Second leg pair, ' + treatment + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')

    xdata = np.array(third_leg_speed)[idx3]
    ydata = np.array(third_step)[idx3]
    pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(xdata,ydata, color = 'gold', linestyle = '', marker = shape, alpha=0.4,label='Third leg pair, ' + treatment + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')

#plt.legend()
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#plt.xlim([0,400])
#plt.ylim([0,200])
#axes.set_ylabel('Step Amplitude ('r'$\mu$m)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Figures/Plots/Byleg_steplength_vs_speed.pdf')

#fig, axes = plt.subplots()

    xdata = np.array(loc_legs_speed)[idx]
    ydata = np.array(loc_legs_step)[idx]
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
    fig.add_subplot(2, 3, count+1)
    plt.plot(xdata,ydata, color = colors[count], linestyle = '', marker = '.', alpha=0.7, label= treatment + ', ' r'$\rho$ = ' + corrtowrite + ', ' + sig)
    plt.plot(np.linspace(2,10),func(np.linspace(2,10),*pars), color = colors[count])
    plt.legend(fontsize=10)
    plt.xlim([0,15])
    #axes[count].set_yticks(fontname='Georgia',fontsize=13)
    #axes[count].set_xticks(fontname='Georgia',fontsize=13)
    plt.ylabel('Step Amplitude', fontname='Georgia', fontsize=13)
    plt.xlabel('Speed', fontname='Georgia', fontsize=13)
plt.savefig(upperdir+'/Figures/Plots/steplength_vs_speed.pdf')

# speed vs period
fig=plt.figure(figsize=(10,7))
count = -1
for treatment in alldata:
    count += 1
    
    first_leg_speed = list(map(float,alldata[treatment]['R1_stride_speed'].tolist()))
    first_leg_speed.extend(list(map(float,alldata[treatment]['L1_stride_speed'].tolist())))
    second_leg_speed = list(map(float,alldata[treatment]['R2_stride_speed'].tolist()))
    second_leg_speed.extend(list(map(float,alldata[treatment]['L2_stride_speed'].tolist())))
    third_leg_speed = list(map(float,alldata[treatment]['R3_stride_speed'].tolist()))
    third_leg_speed.extend(list(map(float,alldata[treatment]['L3_stride_speed'].tolist())))
    loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

    first_period = list(map(float,alldata[treatment]['R1_period'].tolist()))
    first_period.extend(list(map(float,alldata[treatment]['L1_period'].tolist())))
    idx1 = (np.array(first_period) > 0) & (np.array(first_leg_speed) > 1)
    
    second_period = list(map(float,alldata[treatment]['R2_period'].tolist()))
    second_period.extend(list(map(float,alldata[treatment]['L2_period'].tolist())))
    idx2 = (np.array(second_period) > 0) & (np.array(second_leg_speed) > 1)
    
    third_period = list(map(float,alldata[treatment]['R3_period'].tolist()))
    third_period.extend(list(map(float,alldata[treatment]['L3_period'].tolist())))
    idx3 = (np.array(third_period) > 0) & (np.array(third_leg_speed) > 1)
    
    loc_legs_period = first_period + second_period + third_period
    idx = (np.array(loc_legs_period) > 0) & (np.array(loc_legs_speed) > 1)
    
    
    #xdata = np.array(first_leg_speed)[idx1]
    #ydata = np.array(first_period)[idx1]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(np.array(first_leg_speed)[idx1],np.array(first_period)[idx1], color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
    #
    #xdata = np.array(second_leg_speed)[idx2]
    #ydata = np.array(second_period)[idx2]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(second_leg_speed,second_period, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' +  r'$\rho$ = ' + corrtowrite + ', ' + sig)
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
    #
    #xdata = np.array(third_leg_speed)[idx3]
    #ydata = np.array(third_period)[idx3]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(third_leg_speed,third_period, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, '  + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
    #
    #plt.legend()
    #plt.yticks(fontname='Georgia',fontsize=11)
    #plt.xticks(fontname='Georgia',fontsize=11)
    ##plt.xlim([0,400])
    ##plt.ylim([0,2.5])
    #axes.set_ylabel('Period ('r's)', fontname='Georgia', fontsize=13)
    #axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
    #plt.savefig(upperdir+'/Figures/Plots/Byleg_period_vs_speed.pdf')

    xdata = np.array(loc_legs_speed)[idx]
    ydata = np.array(loc_legs_period)[idx]
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
    fig.add_subplot(2, 3, count+1)
    plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_period)[idx], color = colors[count], linestyle = '', marker = '.', alpha=0.7, label= treatment + ',' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'purple')
#
    plt.legend(fontsize=10)
##plt.xlim([0,400])
##plt.ylim([0,2.5])
    plt.yticks(fontname='Georgia',fontsize=13)
    plt.xticks(fontname='Georgia',fontsize=13)
    plt.ylabel('Period (s)', fontname='Georgia', fontsize=18)
    plt.xlabel(' ', fontname='Georgia', fontsize=13)
plt.savefig(upperdir+'/Figures/Plots/Period_vs_speed.pdf')


# speed vs stance
fig=plt.figure(figsize=(10,7))
count = -1
for treatment in alldata:
    count += 1
    
    first_leg_speed = list(map(float,alldata[treatment]['R1_stride_speed'].tolist()))
    first_leg_speed.extend(list(map(float,alldata[treatment]['L1_stride_speed'].tolist())))
    second_leg_speed = list(map(float,alldata[treatment]['R2_stride_speed'].tolist()))
    second_leg_speed.extend(list(map(float,alldata[treatment]['L2_stride_speed'].tolist())))
    third_leg_speed = list(map(float,alldata[treatment]['R3_stride_speed'].tolist()))
    third_leg_speed.extend(list(map(float,alldata[treatment]['L3_stride_speed'].tolist())))
    loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed

    first_stance = list(map(float,alldata[treatment]['R1_stance'].tolist()))
    first_stance.extend(list(map(float,alldata[treatment]['L1_stance'].tolist())))
    idx1 = (np.array(first_stance) > 0) & (np.array(first_leg_speed) > 2)

    second_stance = list(map(float,alldata[treatment]['R2_stance'].tolist()))
    second_stance.extend(list(map(float,alldata[treatment]['L2_stance'].tolist())))
    idx2 = (np.array(second_stance) > 0) & (np.array(second_leg_speed) > 2)
    
    third_stance = list(map(float,alldata[treatment]['R3_stance'].tolist()))
    third_stance.extend(list(map(float,alldata[treatment]['L3_stance'].tolist())))
    idx3 = (np.array(third_stance) > 0) & (np.array(third_leg_speed) > 2)

    loc_legs_stance = first_stance + second_stance + third_stance
    idx = (np.array(loc_legs_stance) > 0) & (np.array(loc_legs_speed) > 0)


    #xdata = np.array(first_leg_speed)[idx1]
    #ydata = np.array(first_stance)[idx1]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(first_leg_speed,first_stance, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
    #
    #xdata = np.array(second_leg_speed)[idx2]
    #ydata = np.array(second_stance)[idx2]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0, 0, 0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
    #plt.plot(second_leg_speed,second_stance, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #
    #xdata = np.array(third_leg_speed)[idx3]
    #ydata = np.array(third_stance)[idx3]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
    #plt.plot(third_leg_speed,third_stance, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #
    ##plt.xlim([0,400])
    ##plt.ylim([0,2.5])
    #
    #plt.legend()
    #plt.yticks(fontname='Georgia',fontsize=11)
    #plt.xticks(fontname='Georgia',fontsize=11)
    #axes.set_ylabel('Stance duration ('r's)', fontname='Georgia', fontsize=13)
    #axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
    #plt.savefig(upperdir+'/Figures/Plots/Byleg_stance_vs_speed.pdf')

    xdata = np.array(loc_legs_speed)[idx]
    ydata = np.array(loc_legs_stance)[idx]
    pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'blue')
    fig.add_subplot(2, 3, count+1)
    plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_stance)[idx], color = colors[count], linestyle = '', marker = '.', alpha=0.7, label= treatment + ', ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
#
    plt.legend(fontsize=10)
    plt.yticks(fontname='Georgia',fontsize=13)
    plt.xticks(fontname='Georgia',fontsize=13)
    plt.ylabel('Stance duration (s)', fontname='Georgia', fontsize=13)
    plt.xlabel('Speed', fontname='Georgia', fontsize=13)
plt.savefig(upperdir+'/Figures/Plots/Stance_vs_speed.pdf')

#speed vs swing
fig=plt.figure(figsize=(10,7))
count = -1
for treatment in alldata:
    count += 1
    
    first_leg_speed = list(map(float,alldata[treatment]['R1_stride_speed'].tolist()))
    first_leg_speed.extend(list(map(float,alldata[treatment]['L1_stride_speed'].tolist())))
    second_leg_speed = list(map(float,alldata[treatment]['R2_stride_speed'].tolist()))
    second_leg_speed.extend(list(map(float,alldata[treatment]['L2_stride_speed'].tolist())))
    third_leg_speed = list(map(float,alldata[treatment]['R3_stride_speed'].tolist()))
    third_leg_speed.extend(list(map(float,alldata[treatment]['L3_stride_speed'].tolist())))
    loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed
    
    first_swing = list(map(float,alldata[treatment]['R1_swing'].tolist()))
    first_swing.extend(list(map(float,alldata[treatment]['L1_swing'].tolist())))
    idx1 = (np.array(first_swing) > 0) & (np.array(first_leg_speed) > 2)

    second_swing = list(map(float,alldata[treatment]['R2_swing'].tolist()))
    second_swing.extend(list(map(float,alldata[treatment]['L2_swing'].tolist())))
    idx2 = (np.array(second_swing) > 0) & (np.array(second_leg_speed) > 2)

    third_swing = list(map(float,alldata[treatment]['R3_swing'].tolist()))
    third_swing.extend(list(map(float,alldata[treatment]['L3_swing'].tolist())))
    idx3 = (np.array(third_swing) > 0) & (np.array(third_leg_speed) > 2)
    
    loc_legs_swing = first_swing + second_swing + third_swing
    idx = (np.array(loc_legs_swing) > 0) & (np.array(loc_legs_speed) > 2)

    #fig, axes = plt.subplots()
    #xdata = np.array(first_leg_speed)[idx1]
    #ydata = np.array(first_swing)[idx1]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    #plt.plot(first_leg_speed,first_swing, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
    #
    #xdata = np.array(second_leg_speed)[idx2]
    #ydata = np.array(second_swing)[idx2]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0, 0, 0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
    #plt.plot(second_leg_speed,second_swing, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #
    #xdata = np.array(third_leg_speed)[idx3]
    #ydata = np.array(third_swing)[idx3]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #corr = scipy.stats.spearmanr(xdata,ydata)
    #corrtowrite = str(round(corr[0],2))
    #if corr[1] < 0.05:
    #    sig = 'p < 0.05'
    #    if corr[1] < 0.01:
    #        sig = 'p < 0.01'
    #    if corr[1] < 0.001:
    #        sig = 'p < 0.001'
    #else:
    #    sig = 'p = ' + str(round(corr[1],2))
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
    #plt.plot(third_leg_speed,third_swing, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #
    #plt.legend()
    ##plt.xlim([0,20])
    ##plt.ylim([0,2.5])
    #plt.yticks(fontname='Georgia',fontsize=11)
    #plt.xticks(fontname='Georgia',fontsize=11)
    #axes.set_ylabel('Swing duration (s)', fontname='Georgia', fontsize=13)
    #axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
    #plt.savefig(upperdir+'/Figures/Plots/Byleg_swing_vs_speed.pdf')
    
    
    xdata = np.array(loc_legs_speed)[idx]
    ydata = np.array(loc_legs_swing)[idx]
    pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
    fig.add_subplot(2, 3, count+1)
    plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_swing)[idx], color = colors[count], linestyle = '', marker = '.', alpha=0.4, label= treatment + ', ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)

    plt.legend()
    plt.yticks(fontname='Georgia',fontsize=11)
    plt.xticks(fontname='Georgia',fontsize=11)
    plt.ylabel('Swing duration (s)', fontname='Georgia', fontsize=13)
    plt.xlabel(' ', fontname='Georgia', fontsize=13)
plt.savefig(upperdir+'/Figures/Plots/Swing_vs_speed.pdf')

fig=plt.figure(figsize=(10,7))
count = -1
for treatment in alldata:
    count += 1
    
    first_leg_speed = list(map(float,alldata[treatment]['R1_stride_speed'].tolist()))
    first_leg_speed.extend(list(map(float,alldata[treatment]['L1_stride_speed'].tolist())))
    second_leg_speed = list(map(float,alldata[treatment]['R2_stride_speed'].tolist()))
    second_leg_speed.extend(list(map(float,alldata[treatment]['L2_stride_speed'].tolist())))
    third_leg_speed = list(map(float,alldata[treatment]['R3_stride_speed'].tolist()))
    third_leg_speed.extend(list(map(float,alldata[treatment]['L3_stride_speed'].tolist())))
    loc_legs_speed = first_leg_speed + second_leg_speed + third_leg_speed
    
    first_swing = list(map(float,alldata[treatment]['R1_swing'].tolist()))
    first_swing.extend(list(map(float,alldata[treatment]['L1_swing'].tolist())))
    idx1 = (np.array(first_swing) > 0) & (np.array(first_leg_speed) > 2)

    second_swing = list(map(float,alldata[treatment]['R2_swing'].tolist()))
    second_swing.extend(list(map(float,alldata[treatment]['L2_swing'].tolist())))
    idx2 = (np.array(second_swing) > 0) & (np.array(second_leg_speed) > 2)

    third_swing = list(map(float,alldata[treatment]['R3_swing'].tolist()))
    third_swing.extend(list(map(float,alldata[treatment]['L3_swing'].tolist())))
    idx3 = (np.array(third_swing) > 0) & (np.array(third_leg_speed) < 13)
    
    loc_legs_swing = first_swing + second_swing + third_swing
    idx = (np.array(loc_legs_swing) > 0) & (np.array(loc_legs_speed) > 2)
    
    first_stance = list(map(float,alldata[treatment]['R1_stance'].tolist()))
    first_stance.extend(list(map(float,alldata[treatment]['L1_stance'].tolist())))
    idx1 = (np.array(first_stance) > 0) & (np.array(first_leg_speed) > 2)

    second_stance = list(map(float,alldata[treatment]['R2_stance'].tolist()))
    second_stance.extend(list(map(float,alldata[treatment]['L2_stance'].tolist())))
    idx2 = (np.array(second_stance) > 0) & (np.array(second_leg_speed) > 2)
    
    third_stance = list(map(float,alldata[treatment]['R3_stance'].tolist()))
    third_stance.extend(list(map(float,alldata[treatment]['L3_stance'].tolist())))
    idx3 = (np.array(third_stance) > 0) & (np.array(third_leg_speed) > 2)

    loc_legs_stance = first_stance + second_stance + third_stance
    idx = (np.array(loc_legs_stance) > 0) & (np.array(loc_legs_speed) < 13)

    xdata = np.array(loc_legs_speed)[idx]
    ydata = np.array(loc_legs_swing)[idx]
    #pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    #residuals = ydata - func(xdata, *pars)
    #ss_res = np.sum(residuals**2)
    #ss_tot = np.sum((ydata-np.mean(ydata))**2)
    #r_squared = 1 - (ss_res / ss_tot)
    #corrtowrite = str(round(r_squared,2))
    #plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
    fig.add_subplot(2, 3, count+1)
    plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_swing)[idx], color = 'red', linestyle = '', marker = '.', alpha=0.8, label= treatment + ', ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    xdata = np.array(loc_legs_speed)[idx]
    ydata = np.array(loc_legs_stance)[idx]
    #print(np.mean(ydata),np.std(ydata))
    pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
    ##residuals = ydata - func(xdata, *pars)
    ##ss_res = np.sum(residuals**2)
    ##ss_tot = np.sum((ydata-np.mean(ydata))**2)
    ##r_squared = 1 - (ss_res / ss_tot)
    ##corrtowrite_stance = str(round(r_squared,2))
    ##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'blue')
    corr = scipy.stats.spearmanr(xdata,ydata)
    corrtowrite = str(round(corr[0],2))
    if corr[1] < 0.05:
        sig = 'p < 0.05'
        if corr[1] < 0.01:
            sig = 'p < 0.01'
        if corr[1] < 0.001:
            sig = 'p < 0.001'
    else:
        sig = 'p = ' + str(round(corr[1],2))
    plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_stance)[idx], color = 'blue', linestyle = '', marker = '.', alpha=0.7, label= treatment + ', ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
    #
    plt.legend(fontsize=10)
    plt.yticks(fontname='Georgia',fontsize=13)
    plt.xticks(fontname='Georgia',fontsize=13)
    plt.ylabel('Phase duration (s)', fontname='Georgia', fontsize=18)
    plt.xlabel(' ', fontname='Georgia', fontsize=13)
plt.savefig(upperdir+'/Figures/Plots/Swingstance_vs_speed.pdf')
