#!/usr/bin/python3

###########################################################################
import os
import requests
import sys
import pandas as pd
import ffmpeg
from datetime import datetime
from colorama import Fore, Back, Style
import os
import shutil
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
    
#this does the actual work of cutting the video to the time of interest and converting it to a format that ImageJ can read
#this uses the ffmpeg bindings, which can be a little finnicky. This script was written with python3 and Debian Linux, might be different on other OS's
#Note: the trim function only seems to work with a start time and a duration, not a start and end time. Seems to be a known bug
def extract_vid(video, start_time, duration):
	start_time_final=start_time.replace(":","-")
	out_video = video.split(".")[0] + "_" + start_time_final + ".avi"
	tmp_out = "/tmp/tempffmpeg.avi"
	print("Now converting: " + video.split(".")[0] + "_" + start_time + ".avi")
	(
	ffmpeg.input(video, ss = start_time, t = duration, an=None)
	.output(tmp_out,  format = "avi", codec = "rawvideo")
	.run(overwrite_output=True)
	
	)
	shutil.move(tmp_out, out_video)
	#os.rename(out_video, video.split(".")[0] + "_" +start_time + ".avi")
	print(out_video)
	
	return
###########################################################################
#The user gets a couple options here. You can either enter in a single video and its assorted info, or you can read in a .csv file that lists all the videos and their timecodes
#You can also quit the script or contniue coding more videos.
basedir = "/media/eebrandt/Cavia/UChicago/NYAS_Flies/sandpaper/data/"
#basedir = "/media/eebrandt/Cavia/UChicago/Living_Machines_Collab/data/"
#basedir = "/home/eebrandt/projects/UChicago/fly_walking/sandpaper/data/"
#basedir = "/home/eebrandt/projects/UChicago/Living_Machines_Collab/data/"
one_or_all = input(Fore.CYAN + Style.BRIGHT + "All in input file (file/f), first in input file (file1/f1) single video (single/s) or exit (quit/q)? ")

#spreadsheetID = '1pDPRbwEd5v_i9nuzeyDEjoCSlntAVpFlRHMjrGzDEUI'
#sheetID = '694749607' 
spreadsheetID = '14np4s690fw7UYScYxyxFJoBc7tl1E7RwvqevCYGaSlI'   
sheetID = '519744848'

while one_or_all != exit:
	if one_or_all == "single" or one_or_all == "s":
		#species = input("Please input species. ")
		#group = input("Please enter group. ")
		ID = input("Please input ID. ")
		video = input("Please input video name. ")
		videoname = basedir + "videos" + "/" + ID + '/'+ video + '.mp4'
		start_time = input("Please input start time. ")
		duration = input("Please input duration. ")
		
		extract_vid(videoname, start_time, duration)
		
		one_or_all = input(Fore.CYAN + Style.BRIGHT + "Input file (file), single video (single) or exit (exit)? ")
	#reads through the .csv file to find each video that we'll be converting, does a little jiggery-pokery with the datetime format, and sends the appropriate data to the function that actually performs the conversion
	elif one_or_all == "file" or one_or_all == "f":
		filepath = getGoogleSheet(spreadsheetID, sheetID, basedir, 'video_timecodes.csv')
		vid_times = pd.read_csv(basedir + "video_timecodes.csv")
		vid_times = vid_times.drop(vid_times[vid_times["done"] == "yes"].index)
		vid_times = vid_times.dropna(subset=["time1"])
		vid_times = vid_times.dropna(subset=["time2"])
		for index, row in vid_times.iterrows():
			species = row['species']
			ID = row['ID']
			group = row['video'][0]
			video = row['video']
			start_time = row['time1']
			end_time = row['time2']
			start_timeobj = datetime.strptime(start_time, '%H:%M:%S')
			end_timeobj = datetime.strptime(end_time, '%H:%M:%S')
			duration = end_timeobj - start_timeobj
			videoname = basedir + "videos" + "/" + ID +'/' + video +'.mp4'
			print(videoname)
			
			extract_vid(videoname, start_time, duration)
			
		one_or_all = input(Fore.CYAN + Style.BRIGHT + "Input entire file (file/f), first file entry (file1/f1), single video (single/s) or exit (quit/q)? ")
	elif one_or_all == "file1" or one_or_all =="f1":
		filepath = getGoogleSeet(spreadsheetID, sheetID, basedir, 'video_timecodes.csv')
		vid_times = pd.read_csv(basedir + "video_timecodes.csv")
		vid_times = vid_times.drop(vid_times[vid_times["done"] == "yes"].index)
		row = vid_times.iloc[0]
		
		#print(row)
		species = row['species']
		ID = row['ID']
		#group = row['video'][0]
		video = row['video']
		start_time = row['time1']
		end_time = row['time2']
		start_timeobj = datetime.strptime(start_time, '%H:%M:%S')
		end_timeobj = datetime.strptime(end_time, '%H:%M:%S')
		duration = end_timeobj - start_timeobj
		videoname = basedir + "videos" + "/" + ID + "/" + video +'.mp4'
		#print(videoname)
		#print(duration)
		
		extract_vid(videoname, start_time, duration)
		
		one_or_all = input(Fore.CYAN + Style.BRIGHT + "Input entire file (file/f), first entry of file (file1/f1) single video (single/s) or exit (quit/q)? ")
		
	elif one_or_all == "quit" or one_or_all == "q":
		quit()		
		
	else:
		one_or_all = input(Fore.RED +Style.BRIGHT+ "Unrecognized input.\n" + Fore.CYAN +"Input entire file (file/f), first entry of file (file1/f1) single video (single/s) or exit (quit/q)? ")
