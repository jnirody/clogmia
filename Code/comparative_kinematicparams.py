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
from scipy import stats

# This makes Figure 5a

####################################################
def func(x, a, b, c):
    return a*x**(b)+c
####################################################

condition = 'sandpaper'

upperdir = '/'.join(os.getcwd().split('/')[:-1])
files = glob.glob(upperdir + '/Data/ForAnalysis/' + condition + '/' + 'compiled*.csv')
alldata = {}
loc_legs_speed = {}
loc_legs_step = {}
loc_legs_swing = {}
loc_legs_stance = {}
loc_legs_period = {}
loc_legs_pullback = {}

for file in files:
    treatment = file.split('/')[-1].split('_')[-1][:-4]
    alldata[treatment] = pd.read_csv(file)
treatments = ['glass','g150','g100','g60','g24']

count = -1
for treatment in alldata:
# Look at how each leg pair correlates with speed
    count += 1
    first_leg_speed = list(map(float,alldata[treatment]['R1_stride_speed'].tolist()))
    first_leg_speed.extend(list(map(float,alldata[treatment]['L1_stride_speed'].tolist())))
    second_leg_speed = list(map(float,alldata[treatment]['R2_stride_speed'].tolist()))
    second_leg_speed.extend(list(map(float,alldata[treatment]['L2_stride_speed'].tolist())))
    third_leg_speed = list(map(float,alldata[treatment]['R3_stride_speed'].tolist()))
    third_leg_speed.extend(list(map(float,alldata[treatment]['L3_stride_speed'].tolist())))

    loc_legs_speed[treatment] = first_leg_speed + second_leg_speed + third_leg_speed
    #print(np.mean(loc_legs_speed2),np.std(loc_legs_speed2))
    
    first_pullback = list(map(float,alldata[treatment]['R1_pull_back'].tolist()))
    first_pullback.extend(list(map(float,alldata[treatment]['L1_pull_back'].tolist())))
    idx1 = (np.array(first_pullback) > 0) & (np.array(first_leg_speed) > 2)

    second_pullback = list(map(float,alldata[treatment]['R2_pull_back'].tolist()))
    second_pullback.extend(list(map(float,alldata[treatment]['L2_pull_back'].tolist())))
    idx2 = (np.array(second_pullback) > 0) & (np.array(second_leg_speed) > 2)

    third_pullback = list(map(float,alldata[treatment]['R3_pull_back'].tolist()))
    third_pullback.extend(list(map(float,alldata[treatment]['L3_pull_back'].tolist())))
    idx3 = (np.array(third_pullback) > 0) & (np.array(third_leg_speed) > 2)
    
    loc_legs_pullback[treatment] = first_pullback + second_pullback + third_pullback

    first_step = list(map(float,alldata[treatment]['R1_step_length'].tolist()))
    first_step.extend(list(map(float,alldata[treatment]['L1_step_length'].tolist())))
    idx1 = (np.array(first_step) > 0)

    second_step = list(map(float,alldata[treatment]['R2_step_length'].tolist()))
    second_step.extend(list(map(float,alldata[treatment]['L2_step_length'].tolist())))
    idx2 = (np.array(second_step) > 0)

    third_step = list(map(float,alldata[treatment]['R3_step_length'].tolist()))
    third_step.extend(list(map(float,alldata[treatment]['L3_step_length'].tolist())))
    idx3 = (np.array(third_step) > 0)

    loc_legs_step[treatment] = first_step + second_step + third_step

    first_period = list(map(float,alldata[treatment]['R1_period'].tolist()))
    first_period.extend(list(map(float,alldata[treatment]['L1_period'].tolist())))
    idx1 = (np.array(first_period) > 0) & (np.array(first_leg_speed) > 1)
    
    second_period = list(map(float,alldata[treatment]['R2_period'].tolist()))
    second_period.extend(list(map(float,alldata[treatment]['L2_period'].tolist())))
    idx2 = (np.array(second_period) > 0) & (np.array(second_leg_speed) > 1)
    
    third_period = list(map(float,alldata[treatment]['R3_period'].tolist()))
    third_period.extend(list(map(float,alldata[treatment]['L3_period'].tolist())))
    idx3 = (np.array(third_period) > 0) & (np.array(third_leg_speed) > 1)
    
    loc_legs_period[treatment] = first_period + second_period + third_period

    first_stance = list(map(float,alldata[treatment]['R1_stance'].tolist()))
    first_stance.extend(list(map(float,alldata[treatment]['L1_stance'].tolist())))
    idx1 = (np.array(first_stance) > 0) & (np.array(first_leg_speed) > 2)

    second_stance = list(map(float,alldata[treatment]['R2_stance'].tolist()))
    second_stance.extend(list(map(float,alldata[treatment]['L2_stance'].tolist())))
    idx2 = (np.array(second_stance) > 0) & (np.array(second_leg_speed) > 2)
    
    third_stance = list(map(float,alldata[treatment]['R3_stance'].tolist()))
    third_stance.extend(list(map(float,alldata[treatment]['L3_stance'].tolist())))
    idx3 = (np.array(third_stance) > 0) & (np.array(third_leg_speed) > 2)

    loc_legs_stance[treatment] = first_stance + second_stance + third_stance
    
    
    first_swing = list(map(float,alldata[treatment]['R1_swing'].tolist()))
    first_swing.extend(list(map(float,alldata[treatment]['L1_swing'].tolist())))
    idx1 = (np.array(first_swing) > 0) & (np.array(first_leg_speed) > 2)

    second_swing = list(map(float,alldata[treatment]['R2_swing'].tolist()))
    second_swing.extend(list(map(float,alldata[treatment]['L2_swing'].tolist())))
    idx2 = (np.array(second_swing) > 0) & (np.array(second_leg_speed) > 2)

    third_swing = list(map(float,alldata[treatment]['R3_swing'].tolist()))
    third_swing.extend(list(map(float,alldata[treatment]['L3_swing'].tolist())))
    idx3 = (np.array(third_swing) > 0) & (np.array(third_leg_speed) > 2)
    
    loc_legs_swing[treatment] = first_swing + second_swing + third_swing

fig, axes = plt.subplots(2,1)
colors = ['orange','blue','green','red','yellow']

df = []
for treatment in treatments:
    print(treatment)
    df.append(loc_legs_step[treatment])
ax = sns.violinplot(data=df,orient='h',ax=axes[0], palette=colors, cut=0)
#for l in ax.lines:
#    l.set_linestyle('--')
#    l.set_linewidth(1)    l.set_color('white')
axes[0].set_yticks([])
axes[0].set_xlabel('Step Amplitude (mm)',fontname='Georgia', fontsize=13)

for treatment in treatments:
    t_stat, p_val = stats.ttest_ind(loc_legs_step[treatment], loc_legs_step['glass'], nan_policy='omit')
    print(np.nanmean(loc_legs_step[treatment]), np.nanstd(loc_legs_step[treatment]), t_stat, p_val, treatment)

df = []
for treatment in treatments:
    df.append(loc_legs_period[treatment])
ax = sns.violinplot(data=df,orient='h', ax=axes[1],palette=colors, cut = 0)
#for l in ax.lines:
#    l.set_linestyle('--')
#    l.set_linewidth(1)
#    l.set_color('white')
axes[1].set_yticks([])
axes[1].set_xlabel('Period (s)',fontname='Georgia', fontsize=13)

for treatment in treatments:
    t_stat, p_val = stats.ttest_ind(loc_legs_period[treatment], loc_legs_period['glass'], nan_policy='omit')
    print(np.nanmean(loc_legs_period[treatment]), np.nanstd(loc_legs_period[treatment]), t_stat, p_val, treatment)
#
#df = []
#for treatment in treatments:
#    df.append(loc_legs_swing[treatment])
#ax = sns.violinplot(data=df,orient='h',inner='quartiles',ax=axes[1][0],palette=colors)
#for l in ax.lines:
#    l.set_linestyle('--')
#    l.set_linewidth(1)
#    l.set_color('yellow')
#axes[1][0].set_yticks([])
#axes[1][0].set_xlabel('Swing (s)',fontname='Georgia', fontsize=13)
#
#df = []
#for treatment in treatments:
#    df.append(loc_legs_stance[treatment])
#ax = sns.violinplot(data=df,orient='h',inner='quartiles',ax=axes[1][1],palette=colors)
#for l in ax.lines:
#    l.set_linestyle('--')
#    l.set_linewidth(1)
#    l.set_color('yellow')
#axes[1][1].set_yticks([])
#axes[1][1].set_xlabel('Stance (s)',fontname='Georgia', fontsize=13)

plt.tight_layout()
plt.savefig(upperdir + '/Figures/' + 'compare_kinematics_by_treatment.pdf')

#

#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_step)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_step)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[1, -0.1, 1], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_step)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_step)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite4 = str(round(r_squared,2))
#plt.plot(xdata,ydata, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.legend()
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#plt.xlim([0,400])
#plt.ylim([0,200])
#axes.set_ylabel('Step Amplitude ('r'$\mu$m)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_steplength_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_step)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,6,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite = str(round(r_squared,2))
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
#plt.plot(xdata,ydata, color = 'green', linestyle = '', marker = '.', alpha=0.7, label= r'$\rho$ = ' + corrtowrite + ', ' + sig)
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,200])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Step Amplitude ('r'$\mu$m)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_steplength_vs_speed.pdf')
#
## speed vs period
#first_period = list(map(float,alldata['R1_period'].tolist()))
#first_period.extend(list(map(float,alldata['L1_period'].tolist())))
#idx1 = (np.array(first_period) > 0) & (np.array(first_leg_speed) > 20)
#
#second_period = list(map(float,alldata['R2_period'].tolist()))
#second_period.extend(list(map(float,alldata['L2_period'].tolist())))
#idx2 = (np.array(second_period) > 0) & (np.array(second_leg_speed) > 20)
#
#third_period = list(map(float,alldata['R3_period'].tolist()))
#third_period.extend(list(map(float,alldata['L3_period'].tolist())))
#idx3 = (np.array(third_period) > 0) & (np.array(third_leg_speed) > 20)
#
#back_period = list(map(float,alldata['R4_period'].tolist()))
#back_period.extend(list(map(float,alldata['L4_period'].tolist())))
#idx4 = (np.array(back_period) > 0) & (np.array(back_leg_speed) > 20)
#
#loc_legs_period = first_period + second_period + third_period
#idx = (np.array(loc_legs_period) > 0) & (np.array(loc_legs_speed) > 20)
#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_period)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(np.array(first_leg_speed)[idx1],np.array(first_period)[idx1], color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_period)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[1, -0.1, 1], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(second_leg_speed,second_period, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_period)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(third_leg_speed,third_period, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_period)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite4 = str(round(r_squared,2))
#plt.plot(back_leg_speed,back_period, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.legend()
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#axes.set_ylabel('Period ('r's)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_period_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_period)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite = str(round(r_squared,2))
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
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_period)[idx], color = 'purple', linestyle = '', marker = '.', alpha=0.7, label= r'$\rho$ = ' + corrtowrite + ', ' + sig)
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'purple')
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Period (s)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_period_vs_speed.pdf')
#
#
## speed vs stance
#first_stance = list(map(float,alldata['R1_stance'].tolist()))
#first_stance.extend(list(map(float,alldata['L1_stance'].tolist())))
#idx1 = (np.array(first_stance) > 0) & (np.array(first_leg_speed) > 20)
#
#second_stance = list(map(float,alldata['R2_stance'].tolist()))
#second_stance.extend(list(map(float,alldata['L2_stance'].tolist())))
#idx2 = (np.array(second_stance) > 0) & (np.array(second_leg_speed) > 20)
#
#third_stance = list(map(float,alldata['R3_stance'].tolist()))
#third_stance.extend(list(map(float,alldata['L3_stance'].tolist())))
#idx3 = (np.array(third_stance) > 0) & (np.array(third_leg_speed) > 20)
#
#back_stance = list(map(float,alldata['R4_stance'].tolist()))
#back_stance.extend(list(map(float,alldata['L4_stance'].tolist())))
#idx4 = (np.array(back_stance) > 0) & (np.array(back_leg_speed) > 20)
#
#loc_legs_stance = first_stance + second_stance + third_stance
#idx = (np.array(loc_legs_stance) > 0) & (np.array(loc_legs_speed) > 20)
#
#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_stance)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(first_leg_speed,first_stance, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_stance)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0, 0, 0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#plt.plot(second_leg_speed,second_stance, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_stance)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#plt.plot(third_leg_speed,third_stance, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_stance)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite4 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
#plt.plot(back_leg_speed,back_stance, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#
#plt.legend()
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#axes.set_ylabel('Stance duration ('r's)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_stance_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_stance)[idx]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite_stance = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'blue')
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_stance)[idx], color = 'blue', linestyle = '', marker = '.', alpha=0.7, label= r'$R^2$ = ' + corrtowrite_stance)
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Stance duration (s)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_stance_vs_speed.pdf')
#
##speed vs swing
#first_swing = list(map(float,alldata['R1_swing'].tolist()))
#first_swing.extend(list(map(float,alldata['L1_swing'].tolist())))
#idx1 = (np.array(first_swing) > 0) & (np.array(first_leg_speed) > 20)
#
#second_swing = list(map(float,alldata['R2_swing'].tolist()))
#second_swing.extend(list(map(float,alldata['L2_swing'].tolist())))
#idx2 = (np.array(second_swing) > 0) & (np.array(second_leg_speed) > 20)
#
#third_swing = list(map(float,alldata['R3_swing'].tolist()))
#third_swing.extend(list(map(float,alldata['L3_swing'].tolist())))
#idx3 = (np.array(third_swing) > 0) & (np.array(third_leg_speed) > 20)
#
#back_swing = list(map(float,alldata['R4_swing'].tolist()))
#back_swing.extend(list(map(float,alldata['L4_swing'].tolist())))
#idx4 = (np.array(back_swing) > 0) & (np.array(back_leg_speed) > 20)
#
#loc_legs_swing = first_swing + second_swing + third_swing
#idx = (np.array(loc_legs_swing) > 0) & (np.array(loc_legs_speed) > 20)
#
#fig, axes = plt.subplots()
#xdata = np.array(first_leg_speed)[idx1]
#ydata = np.array(first_swing)[idx1]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite1 = str(round(r_squared,2))
#plt.plot(first_leg_speed,first_swing, color = 'blue', linestyle = '', marker = '.', alpha=0.4,label='First leg pair, ' + r'$R^2$ = ' + corrtowrite1)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='blue')
#
#xdata = np.array(second_leg_speed)[idx2]
#ydata = np.array(second_swing)[idx2]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0, 0, 0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite2 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'green')
#plt.plot(second_leg_speed,second_swing, color = 'green', linestyle = '', marker = '.', alpha=0.4, label='Second leg pair, ' + r'$R^2$ = ' + corrtowrite2)
#
#xdata = np.array(third_leg_speed)[idx3]
#ydata = np.array(third_swing)[idx3]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'gold')
#plt.plot(third_leg_speed,third_swing, color = 'gold', linestyle = '', marker = '.', alpha=0.4,label='Third leg pair, ' + r'$R^2$ = ' + corrtowrite3)
#
#
#xdata = np.array(back_leg_speed)[idx4]
#ydata = np.array(back_swing)[idx4]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite3 = str(round(r_squared,2))
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
#plt.plot(back_leg_speed,back_swing, color = 'red', linestyle = '', marker = '.', alpha=0.4, label='Back leg pair, ' + r'$R^2$ = ' + corrtowrite4)
#plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color='red')
#
#plt.legend()
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#axes.set_ylabel('Swing duration (s)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_byleg_swing_vs_speed.pdf')
#
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_swing)[idx]
#pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
#residuals = ydata - func(xdata, *pars)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata-np.mean(ydata))**2)
#r_squared = 1 - (ss_res / ss_tot)
#corrtowrite = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_swing)[idx], color = 'red', linestyle = '', marker = '.', alpha=0.4, label= r'$R^2$ = ' + corrtowrite)
#
#plt.legend()
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=11)
#plt.xticks(fontname='Georgia',fontsize=11)
#axes.set_ylabel('Swing duration (s)', fontname='Georgia', fontsize=13)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_swing_vs_speed.pdf')
#
#fig, axes = plt.subplots()
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_swing)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'red')
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
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_swing)[idx], color = 'red', linestyle = '', marker = '.', alpha=0.8, label= 'Swing: ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
#xdata = np.array(loc_legs_speed)[idx]
#ydata = np.array(loc_legs_stance)[idx]
##pars, cov = curve_fit(func,xdata,ydata, p0=[0,0,0], maxfev=100000)
##residuals = ydata - func(xdata, *pars)
##ss_res = np.sum(residuals**2)
##ss_tot = np.sum((ydata-np.mean(ydata))**2)
##r_squared = 1 - (ss_res / ss_tot)
##corrtowrite_stance = str(round(r_squared,2))
##plt.plot(np.linspace(50,350),func(np.linspace(50,350),*pars), color = 'blue')
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
#plt.plot(np.array(loc_legs_speed)[idx],np.array(loc_legs_stance)[idx], color = 'blue', linestyle = '', marker = '.', alpha=0.7, label= 'Stance: ' + r'$\rho$ = ' + corrtowrite + ', ' + sig)
#
#plt.legend(fontsize=16)
#plt.xlim([0,400])
#plt.ylim([0,2.5])
#plt.yticks(fontname='Georgia',fontsize=13)
#plt.xticks(fontname='Georgia',fontsize=13)
#axes.set_ylabel('Phase duration (s)', fontname='Georgia', fontsize=18)
#axes.set_xlabel(' ', fontname='Georgia', fontsize=13)
#plt.savefig(upperdir+'/Writeup/Figures/' + stiffness + 'kPa_swingstance_vs_speed.pdf')
