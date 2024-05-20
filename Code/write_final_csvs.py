#!/usr/bin/python3

# writes out csvs for analysis
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
import requests
###########################################################################
def grouper(iterable,n):
    args = [iter(iterable)]*n
    return zip(*args)
###########################################################################
#this function downloads a google sheet based on a given spreadsheet ID and sheet ID
def getGoogleSheet(spreadsheetID, sheetID, outDir, outFile):
  url = f'https://docs.google.com/spreadsheets/d/'+ spreadsheetID +'/gviz/tq?tqx=out:csv&gid='+ sheetID
  response = requests.get(url)
  if response.status_code == 200:
    print("downloading")
    filepath = os.path.join(outDir, outFile)
    with open(filepath, 'wb') as f:
      f.write(response.content)
      print('CSV file saved to: {}'.format(filepath))    
  else:
    print(f'Error downloading Google Sheet: {response.status_code}')
    sys.exit(1)
###########################################################################
condition = 'sandpaper'
spreadsheetID = '14np4s690fw7UYScYxyxFJoBc7tl1E7RwvqevCYGaSlI'   
sheetID = '519744848'

currdir = os.getcwd()
upperdir = '/'.join(currdir.split('/')[:-1])
upperdir = '/home/eebrandt/projects/UChicago/fly_walking/sandpaper/'
directory = upperdir + 'data/tracking/'
print(directory)

if condition == 'ablatement':
    treatments = [f for f in os.listdir(directory) if not f.startswith('.')]
if condition == 'sandpaper':
    treatments = ['sp']
    flies = [f for f in os.listdir(directory) if not f.startswith('.')]
    
filepath = getGoogleSheet(spreadsheetID, sheetID, directory, 'video_timecodes.csv')    
metadata = pd.read_csv(directory + 'metadata.csv')

framerate = 240. # frames/sec

flydata = {} # create a dictionary to then write out from
for tr in treatments:
    if condition == 'ablatement':
        directory = directory + tr + '/'
        flies = [f for f in os.listdir(directory) if not f.startswith('.')]
    for fly in flies:
        files = glob.glob(directory + fly + '/*.csv')
        for file in files:
            vid_file = file.split('/')[-1][:-4] + '.avi'
            print(file)
            resolution = 1./(metadata[metadata['video'] == vid_file]['spatial_calibration'].iloc[0]) # mm / pixel
            treatment = file.split('/')[-1].split('_')[1]
            trial = file.split('/')[-1].split('_')[2]
            video = file.split('/')[-1].split('_')[3]
            timestamp = file.split('/')[-1].split('_')[4][:-4]
            dataframe = pd.read_csv(file)
            grouped = dataframe.groupby('TID')
            max_time = min(len(dataframe[dataframe['TID'] == 7]['t [sec]']),len(dataframe[dataframe['TID'] == 8]['t [sec]']),len(dataframe[dataframe['TID'] == 9]['t [sec]']))
            if fly in flydata.keys():
                if treatment in flydata[fly].keys():
                    if video in flydata[fly][treatment].keys():
                        flydata[fly][treatment][video][timestamp] = {}
                        flydata[fly][treatment][video][timestamp][0] = [trial]
                    else:
                        flydata[fly][treatment][video] = {}
                        flydata[fly][treatment][video][timestamp] = {}
                        flydata[fly][treatment][video][timestamp][0] = [trial, metadata[metadata['video'] == vid_file]['body_length'].iloc[0]]
                else:
                    flydata[fly][treatment] = {}
                    flydata[fly][treatment][video] = {}
                    flydata[fly][treatment][video][timestamp] = {}
                    flydata[fly][treatment][video][timestamp][0] = [trial]
            else:
                flydata[fly] = {}
                flydata[fly][treatment] = {}
                flydata[fly][treatment][video] = {}
                flydata[fly][treatment][video][timestamp] = {}
                flydata[fly][treatment][video][timestamp][0] = [trial]
            to_subtract = 0
            for leg,data in grouped:
                ind = None
                flydata[fly][treatment][video][timestamp][leg] = {}
                if leg == 7:
                    flydata[fly][treatment][video][timestamp][leg]['head_pos'] = []
                    for i in range(max_time-1):
                        hx = data['x [pixel]'].iloc[[i][0]]
                        hy = data['y [pixel]'].iloc[[i][0]]
                        flydata[fly][treatment][video][timestamp][leg]['head_pos'].append((hx,hy))
                elif leg == 8:
                    flydata[fly][treatment][video][timestamp][leg]['center_pos'] = []
                    flydata[fly][treatment][video][timestamp][leg]['dist'] = []
                    flydata[fly][treatment][video][timestamp][leg]['velocity'] = []
                    flydata[fly][treatment][video][timestamp][leg]['num_feet_down'] = []
                    flydata[fly][treatment][video][timestamp][leg]['body_length'] = [ metadata[metadata['video'] == vid_file]['body_length'].iloc[0]]
                    for i in range(max_time):
                        cx1 = data['x [pixel]'].iloc[[i][0]]
                        cy1 = data['y [pixel]'].iloc[[i][0]]
                        flydata[fly][treatment][video][timestamp][leg]['center_pos'].append((cx1,cy1))
                        if i < max_time - 1:
                            cx2 = data['x [pixel]'].iloc[[i+1][0]]
                            cy2 = data['y [pixel]'].iloc[[i+1][0]]
                            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)*resolution
                            flydata[fly][treatment][video][timestamp][leg]['dist'].append(dist)
                            flydata[fly][treatment][video][timestamp][leg]['velocity'].append(dist*framerate/metadata[metadata['video'] == vid_file]['body_length'].iloc[0])
                        else:
                            flydata[fly][treatment][video][timestamp][leg]['dist'].append('0.0')
                            flydata[fly][treatment][video][timestamp][leg]['velocity'].append('0.0')
                elif leg == 9:
                    flydata[fly][treatment][video][timestamp][leg]['tail_pos'] = []
                    for i in range(max_time-1):
                        tx = data['x [pixel]'].iloc[[i][0]]
                        ty = data['y [pixel]'].iloc[[i][0]]
                        flydata[fly][treatment][video][timestamp][leg]['tail_pos'].append((tx,ty))
                elif leg > 9:
                    continue
                else:
                    flydata[fly][treatment][video][timestamp][leg]['liftoff_time'] = []
                    flydata[fly][treatment][video][timestamp][leg]['duty_factor'] = []
                    flydata[fly][treatment][video][timestamp][leg]['period'] = []
                    flydata[fly][treatment][video][timestamp][leg]['step_length'] = []
                    flydata[fly][treatment][video][timestamp][leg]['stride_length'] = []
                    flydata[fly][treatment][video][timestamp][leg]['stride_center_speed'] = []
                    flydata[fly][treatment][video][timestamp][leg]['pull_back'] = []
                    flydata[fly][treatment][video][timestamp][leg]['stance'] = []
                    flydata[fly][treatment][video][timestamp][leg]['swing'] = []
                    flydata[fly][treatment][video][timestamp][leg]['strides'] = []
                    flydata[fly][treatment][video][timestamp][leg]['liftoff_time'] = []
                    flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'] = []
                    flydata[fly][treatment][video][timestamp][leg]['liftoff_loc_bfc'] = []
                    flydata[fly][treatment][video][timestamp][leg]['touchdown_time'] = []
                    flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'] = []
                    flydata[fly][treatment][video][timestamp][leg]['touchdown_loc_bfc'] = []
                    flydata[fly][treatment][video][timestamp][leg]['leg_down'] = [0]*(max_time)
                    flydata[fly][treatment][video][timestamp][leg]['leg_down_loc'] = [0]*(max_time)
                    for i in range(max_time-1):
                        if i in data['t [sec]'].tolist():
                            ind = data.index[data['t [sec]'] == i].tolist()[0] - to_subtract
                            flydata[fly][treatment][video][timestamp][leg]['leg_down'][i] = 1
                            flydata[fly][treatment][video][timestamp][leg]['leg_down_loc'][i] = (data['x [pixel]'].iloc[[ind][0]],data['y [pixel]'].iloc[[ind][0]])
                            if (i-1) not in data['t [sec]'].tolist() and i > 0:
                                flydata[fly][treatment][video][timestamp][leg]['touchdown_time'].append(i)
                                flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'].append((data['x [pixel]'].iloc[[ind][0]],data['y [pixel]'].iloc[[ind][0]]))
                        else:
                            if (i-1) in data['t [sec]'].tolist():
                                ind = data.index[data['t [sec]'] == (i-1)].tolist()[0] - to_subtract
                                flydata[fly][treatment][video][timestamp][leg]['liftoff_time'].append(i-1)
                                flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'].append((data['x [pixel]'].iloc[[ind][0]],data['y [pixel]'].iloc[[ind][0]]))
                    to_subtract += len(data)
            temp_style = [0]*len(flydata[fly][treatment][video][timestamp][1]['leg_down'])
            for k in range(1,7):
                temp_style = [y + z for y,z in zip(flydata[fly][treatment][video][timestamp][k]['leg_down'],temp_style)]
            flydata[fly][treatment][video][timestamp][8]['num_feet_down'] = temp_style
            for treatment in flydata[fly]:
                with open(upperdir + '/data/processed_data/Individual/ByFrame/' + fly + '_' + treatment + '_framebyframe.csv','w') as outfile:
                    fly_writer = csv.writer(outfile, delimiter=',')
                    fly_writer.writerow(['fly','treatment','trial','iteration','body_length','limb_length','frame', 'L1_leg_down', 'L1_leg_loc','R1_leg_down','R1_leg_loc','L2_leg_down','L2_leg_loc', 'R2_leg_down','R2_leg_loc','L3_leg_down', 'L3_leg_loc','R3_leg_down','R3_leg_loc','head_pos', 'center_pos','COM_dist','COM_speed','num_feet_down','tail_pos'])
                    for video in flydata[fly][treatment]:
                        for timestamp in flydata[fly][treatment][video]:
                            row = []
                            for leg in flydata[fly][treatment][video][timestamp]:
                                if leg == 0:
                                    for i in range(len(flydata[fly][treatment][video][timestamp][1]['leg_down'])):
                                        row.append([fly, treatment, video + '_' + timestamp,flydata[fly][treatment][video][timestamp][0][0],metadata[metadata['video'] == vid_file]['body_length'].iloc[0],' ',i])
                                elif leg == 7:
                                    for i in range(len(flydata[fly][treatment][video][timestamp][leg]['head_pos'])):
                                        row[i].extend([flydata[fly][treatment][video][timestamp][leg]['head_pos'][i]])
                                elif leg == 8:
                                    for i in range(len(flydata[fly][treatment][video][timestamp][leg]['center_pos'])):
                                        row[i].extend([flydata[fly][treatment][video][timestamp][leg]['center_pos'][i], flydata[fly][treatment][video][timestamp][leg]['dist'][i],flydata[fly][treatment][video][timestamp][leg]['velocity'][i],flydata[fly][treatment][video][timestamp][leg]['num_feet_down'][i]])
                                elif leg == 9:
                                    for i in range(len(flydata[fly][treatment][video][timestamp][leg]['tail_pos'])):
                                        row[i].extend([flydata[fly][treatment][video][timestamp][leg]['tail_pos'][i]])
                                elif leg > 9:
                                    continue
                                else:
                                    for i in range(len(flydata[fly][treatment][video][timestamp][leg]['leg_down'])):
                                        row[i].extend([flydata[fly][treatment][video][timestamp][leg]['leg_down'][i],flydata[fly][treatment][video][timestamp][leg]['leg_down_loc'][i]])
                            for j in range(len(row)):
                                if len(row[j]) != 25:
                                    continue
                                fly_writer.writerow(row[j])
                           
                            # calculate the values for 'by stride' calculations
                            for leg in flydata[fly][treatment][video][timestamp]:
                                if leg > 0 and leg < 7:
                                    strides = []
                                    if flydata[fly][treatment][video][timestamp][leg]['touchdown_time'][0] < flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][0]:
                                        strides.append(['',flydata[fly][treatment][video][timestamp][leg]['touchdown_time'][0], '','','','','','','','',flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'][0], ''])
                                        flydata[fly][treatment][video][timestamp][leg]['to_shift'] = 1
                                    else:
                                        flydata[fly][treatment][video][timestamp][leg]['to_shift'] = 0
                                    for i in range(len(flydata[fly][treatment][video][timestamp][leg]['liftoff_time'])-1):
                                        flydata[fly][treatment][video][timestamp][leg]['period'].extend([float(flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i+1]-flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i])/framerate])
                                        PEP1x = flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'][i][0]
                                        PEP1y = flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'][i][1]
                                        PEP2x = flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'][i+1][0]
                                        PEP2y = flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'][i+1][1]
                                        AEP1x = flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']][0]
                                        AEP1y = flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']][1]
                                        cpos1  = flydata[fly][treatment][video][timestamp][8]['center_pos'][flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i]]
                                        cpos2  = flydata[fly][treatment][video][timestamp][8]['center_pos'][flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i+1]]
                                        dist = np.sqrt((cpos1[0]-cpos2[0])**2 + (cpos1[1]-cpos2[1])**2)*resolution
                                        flydata[fly][treatment][video][timestamp][leg]['stride_center_speed'].append(dist/(float(flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i+1]-flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i])/framerate))
                                        if i + flydata[fly][treatment][video][timestamp][leg]['to_shift']+1 < len(flydata[fly][treatment][video][timestamp][leg]['touchdown_loc']):
                                            AEP2x = flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']+1][0]
                                            AEP2y = flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']+1][1]
                                            flydata[fly][treatment][video][timestamp][leg]['stride_length'].append(np.sqrt((AEP2x-AEP1x)**2+ (AEP2y-AEP1y)**2)*resolution)
                                        else:
                                            AEP2x = AEP1x
                                            AEP2y = AEP1y
                                            flydata[fly][treatment][video][timestamp][leg]['stride_length'].append(np.mean( flydata[fly][treatment][video][timestamp][leg]['stride_length']))
                                        flydata[fly][treatment][video][timestamp][leg]['pull_back'].append(np.sqrt((PEP2x-AEP1x)**2+(PEP2y-AEP1y)**2)*resolution)
                                        flydata[fly][treatment][video][timestamp][leg]['step_length'].append(np.sqrt((PEP1x-AEP1x)**2+(PEP1y-AEP1y)**2)*resolution)
                                        flydata[fly][treatment][video][timestamp][leg]['swing'].append( float(flydata[fly][treatment][video][timestamp][leg]['touchdown_time'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']] - flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i])/framerate)
                                        flydata[fly][treatment][video][timestamp][leg]['stance'].append(float(flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i+1] - flydata[fly][treatment][video][timestamp][leg]['touchdown_time'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']])/framerate)
                                        flydata[fly][treatment][video][timestamp][leg]['duty_factor'].append(flydata[fly][treatment][video][timestamp][leg]['stance'][i]/float(flydata[fly][treatment][video][timestamp][leg]['period'][i]))
                                        strides.append([flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][i], flydata[fly][treatment][video][timestamp][leg]['touchdown_time'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']],flydata[fly][treatment][video][timestamp][leg]['swing'][i],flydata[fly][treatment][video][timestamp][leg]['stance'][i],flydata[fly][treatment][video][timestamp][leg]['period'][i],flydata[fly][treatment][video][timestamp][leg]['duty_factor'][i],flydata[fly][treatment][video][timestamp][leg]['step_length'][i],flydata[fly][treatment][video][timestamp][leg]['pull_back'][i],flydata[fly][treatment][video][timestamp][leg]['stride_length'][i],flydata[fly][treatment][video][timestamp][leg]['stride_center_speed'][i],flydata[fly][treatment][video][timestamp][leg]['touchdown_loc'][i+flydata[fly][treatment][video][timestamp][leg]['to_shift']],flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'][i]])
                                    strides.append([flydata[fly][treatment][video][timestamp][leg]['liftoff_time'][-1],'', '','','','','','','','','', flydata[fly][treatment][video][timestamp][leg]['liftoff_loc'][-1]])
                                    flydata[fly][treatment][video][timestamp][leg]['strides'] = strides
                            # write individual stride-by-stride csvs to analyse
                with open(upperdir + '/data/processed_data/Individual/ByStride/' + fly + '_' + treatment + '_stridebystride.csv','w') as outfile:
                    fly_writer = csv.writer(outfile, delimiter=',')
                    fly_writer.writerow(['fly','treatment', 'trial', 'iteration', 'body_length','limb_length','L1_swing_start','L1_stance_start','L1_swing','L1_stance', 'L1_period','L1_duty_factor','L1_step_length','L1_pull_back','L1_stride_length','L1_stride_speed','L1_AEP','L1_PEP', 'R1_swing_start','R1_stance_start','R1_swing','R1_stance','R1_period', 'R1_duty_factor','R1_step_length','R1_pull_back', 'R1_stride_length','R1_stride_speed', 'R1_AEP','R1_PEP', 'L2_swing_start','L2_stance_start','L2_swing','L2_stance','L2_period','L2_duty_factor', 'L2_step_length','L2_pull_back','L2_stride_length','L2_stride_speed','L2_AEP','L2_PEP', 'R2_swing_start','R2_stance_start','R2_swing','R2_stance', 'R2_period','R2_duty_factor','R2_step_length','R2_pull_back','R2_stride_length','R2_stride_speed','R2_AEP','R2_PEP', 'L3_swing_start','L3_stance_start','L3_swing','L3_stance', 'L3_period','L3_duty_factor','L3_step_length','L3_pull_back','L3_stride_length','L3_stride_speed','L3_AEP','L3_PEP', 'R3_swing_start','R3_stance_start','R3_swing','R3_stance','R3_period', 'R3_duty_factor','R3_step_length','R3_pull_back','R3_stride_length','R3_stride_speed','R3_AEP','R3_PEP'])
                    for video in flydata[fly][treatment]:
                        for timestamp in flydata[fly][treatment][video]:
                            row = []
                            for leg in flydata[fly][treatment][video][timestamp]:
                                if leg == 0:
                                    longest_run = max(len(flydata[fly][treatment][video][timestamp][1]['strides']),len(flydata[fly][treatment][video][timestamp][2]['strides']),len(flydata[fly][treatment][video][timestamp][3]['strides']),len(flydata[fly][treatment][video][timestamp][4]['strides']),len(flydata[fly][treatment][video][timestamp][5]['strides']),len(flydata[fly][treatment][video][timestamp][6]['strides']))
                                    for i in range(longest_run):
                                        row.append([fly,treatment, video + '_'+ timestamp,flydata[fly][treatment][video][timestamp][0][0], metadata[metadata['video'] == vid_file]['body_length'].iloc[0],resolution])
                                elif leg > 0 and leg < 7:
                                    for i in range(longest_run):
                                        try:
                                            row[i].extend(flydata[fly][treatment][video][timestamp][leg]['strides'][i])
                                        except IndexError:
                                            row[i].extend(['','','','','','','', '', '','', '',''])
                            for j in range(len(row)):
                                fly_writer.writerow(row[j])
fly_average_rows = []
for fly in flydata:
    for treatment in flydata[fly]:
        fly_average_rows.append([fly, treatment])
        body_length = []
        c_body_length = []
        limb_length = []
        cum_swing = []
        cum_stance = []
        cum_period = []
        cum_duty_factor = []
        cum_step_length = []
        cum_pull_back = []
        cum_stride_length = []
        swing = []
        stance = []
        period = []
        duty_factor = []
        step_length = []
        pull_back = []
        stride_length = []
        speed = []
        for video in flydata[fly][treatment]:
            for timestamp in flydata[fly][treatment][video]:
                for leg in flydata[fly][treatment][video][timestamp]:
                    if leg == 0:
                        body_length = flydata[fly][treatment][video][timestamp][8]['body_length']
                        limb_length.extend([' '])
                    elif leg == 7:
                        continue
                    elif leg == 8:
                        # should this be replaced with COM speed?
                        speed.extend([(np.cumsum(flydata[fly][treatment][video][timestamp][leg]['dist'][1:-1])[-1])/(len(flydata[fly][treatment][video][timestamp][leg]['dist'][1:-1])/framerate)])
                    elif leg > 8:
                        continue
                    else:
                        swing.append([flydata[fly][treatment][video][timestamp][leg]['swing']])
                        stance.append([flydata[fly][treatment][video][timestamp][leg]['stance']])
                        period.append([flydata[fly][treatment][video][timestamp][leg]['period']])
                        duty_factor.append([flydata[fly][treatment][video][timestamp][leg]['duty_factor']])
                        step_length.append([flydata[fly][treatment][video][timestamp][leg]['step_length']])
                        pull_back.append([flydata[fly][treatment][video][timestamp][leg]['pull_back']])
                        stride_length.append([flydata[fly][treatment][video][timestamp][leg]['stride_length']])
                        if leg < 7:
                            cum_swing.extend(flydata[fly][treatment][video][timestamp][leg]['swing'])
                            cum_stance.extend(flydata[fly][treatment][video][timestamp][leg]['stance'])
                            cum_period.extend(flydata[fly][treatment][video][timestamp][leg]['period'])
                            cum_duty_factor.extend(flydata[fly][treatment][video][timestamp][leg]['duty_factor'])
                            cum_step_length.extend(flydata[fly][treatment][video][timestamp][leg]['step_length'])
                            cum_pull_back.extend(flydata[fly][treatment][video][timestamp][leg]['pull_back'])
                            cum_stride_length.extend(flydata[fly][treatment][video][timestamp][leg]['stride_length'])
        fly_average_rows[-1].extend([body_length,' ',' ',' '])
        for i in range(6):
            fly_average_rows[-1].extend([np.mean(swing[i]),np.std(swing[i]),np.mean(stance[i]),np.std(stance[i]),np.mean(period[i]), np.std(stance[i]),np.mean(duty_factor[i]),np.std(duty_factor[i]),np.mean(step_length[i]),np.std(step_length[i]),np.mean(pull_back[i]),np.std(pull_back[i]),np.mean(stride_length[i]),np.std(stride_length[i])])
        fly_average_rows[-1].extend([np.mean(cum_swing),np.std(cum_swing), np.mean(cum_stance),np.std(cum_stance), np.mean(cum_period),np.std(cum_period),np.mean(cum_duty_factor), np.std(cum_duty_factor),np.mean(cum_step_length), np.std(cum_step_length),np.mean(cum_pull_back),np.std(cum_pull_back),np.mean(cum_stride_length), np.std(cum_stride_length)])
        fly_average_rows[-1].extend([np.mean(speed),np.std(speed),' '])

with open(upperdir + '/data/processed_data/fly_averages.csv', 'w') as outfile:
    fly_writer = csv.writer(outfile, delimiter=',')
    fly_writer.writerow(['fly','treatment','body_length','body_length_sd','limb_length','limb_length_sd','L1_swing','L1_swing_sd','L1_stance','L1_stance_sd','L1_period','L1_period_sd','L1_duty_factor','L1_duty_factor_sd', 'L1_step_length','L1_step_length_sd','L1_pull_back','L1_pull_back_sd','L1_stride_length','L1_stride_length_sd','R1_swing','R1_swing_sd','R1_stance','R1_stance_sd','R1_period','R1_period_sd','R1_duty_factor','R1_duty_factor_sd', 'R1_step_length','R1_step_length_sd','R1_pull_back','R1_pull_back_sd','R1_stride_length','R1_stride_length_sd','L2_swing', 'L2_swing_sd','L2_stance','L2_stance_sd','L2_period','L2_period_sd','L2_duty_factor','L2_duty_factor_sd', 'L2_step_length','L2_step_length_sd','L2_pull_back','L2_pull_back_sd','L2_stride_length','L2_stride_length_sd','R2_swing', 'R2_swing_sd','R2_stance','R2_stance_sd','R2_period','R2_period_sd','R2_duty_factor','R2_duty_factor_sd', 'R2_step_length','R2_step_length_sd','R2_pull_back','R2_pull_back_sd','R2_stride_length','R2_stride_length_sd','L3_swing', 'L3_swing_sd','L3_stance','L3_stance_sd','L3_period','L3_period_sd','L3_duty_factor','L3_duty_factor_sd', 'L3_step_length','L3_step_length_sd','L3_pull_back','L3_pull_back_sd','L3_stride_length','L3_stride_length_sd','R3_swing', 'R3_swing_sd','R3_stance','R3_stance_sd','R3_period','R3_period_sd','R3_duty_factor','R3_duty_factor_sd', 'R3_step_length','R3_step_length_sd','R3_pull_back','R3_pull_back_sd','R3_stride_length','R3_stride_length_sd','cum_swing','cum_swing_sd','cum_stance','cum_stance_sd', 'cum_period','cum_period_sd', 'cum_duty_factor','cum_duty_factor_sd', 'cum_step_length','cum_step_length_sd', 'cum_pull_back','cum_pull_back_sd','cum_stride_length','cum_stride_length_sd', 'speed','speed_sd','Froude_number'])
    for j in range(len(fly_average_rows)):
        fly_writer.writerow(fly_average_rows[j])

concatdir = upperdir + '/data/processed_data/Individual/ByStride/'
filenames = glob.glob(concatdir + "/*.csv")
sorted_files = {}
big_df = {}

for file in filenames:
    treatment = file.split('/')[-1].split('_')[1]
    if treatment in sorted_files.keys():
        sorted_files[treatment].append(file)
    else:
        sorted_files[treatment] = [file]
        big_df[treatment] = []
        
for treatment in sorted_files:
    for file in sorted_files[treatment]:
        df = pd.read_csv(file, index_col=None, header=0)
        big_df[treatment].append(df)
        
for treatment in big_df:
    frame = pd.concat(big_df[treatment], axis=0, ignore_index=True)
    frame.to_csv(upperdir + 'data/processed_data/' + 'compiled_flies_by_strides_' + treatment + '.csv')

