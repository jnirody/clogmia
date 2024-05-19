Trial tracking, metadata for videos, etc

NOTE: most of the sheets used here will get fed into a script so please do not change column names, add/remove colunms, etc without changing the code.

Clogmia wing ablatement:
	- Google sheet: https://docs.google.com/spreadsheets/d/1zKYECgy26DG9HUBMTRWkpdzwNgYklaDJlNNeK5kXR4s/edit?usp=sharing
		- Tabs: 
			- prelim_wing_ablatement: metadata for initial trials, not to be included in paper
			- wing_ablatement_2: metadata for wing ablatement trials to be included in paper details for each column are found in the tracking protocol
			- video_timecodes: where to cut videos in order to extract walking bouts (>2 continuous strides)
			- ablatement_trial_tracker: Erin's guide to trial, video chopping, tracking progress. Feel free to play with this data, make graphs, whatever. This doesn't get fed into a script.
Clogmia sandpaper:
	- Google sheet: https://docs.google.com/spreadsheets/d/14np4s690fw7UYScYxyxFJoBc7tl1E7RwvqevCYGaSlI/edit?usp=sharing
		Tabs:
			- sandpaper timecodes: same idea as for the wing ablatement trials
			- sandpaper_trial_tracker: same as for wing ablatement trials
			- sandpaper_tracking_metadata: metadata for sandpaper trials, feel free to use to keep track of video chopping/tracking
Guide to folders:
	
Clogmia wing ablatement:
	- contains data and code specific to trials in which fly walking was recorded on glass at varying times before, immediately after, and after wing removal
	- Each fly belonged to one of four different treatments:
			- Treatment A: wings were removed
			- Treatment B: flies were anesthetized and handled but wings were not removed
			- Treatment C: flies were not anesthetized or altered
			- Treatment D: flies were not anesthetized or altered, and only measured twice instead of three times, (only trial 1 and 3, no 2)
	- Regardless of treatment, each fly was measured once as a basline with no manipulation, once one hour post-manipulation (or one hour after first trial in case of group C), and once the following day
	- Example filename: CA22_B_1_2_00-00-02.avi
		- CA22: individual ID of fly (Clogmia albipunctata number 22)
		- B: individual belonged to treatment B
		- 1: first trial (before manipulation)
		- 2: second video captured during this trial
		- 00:00:02 timestamp of specific walking bout

Clogmia sandpaper:
	- contains data and code specific to trials in which flies walked on five different tracks of varying roughness (sandpaper grits)
	- All flies were theoretically exposed to each treatment, presented in a random order (determined by random.org). However, most flies only completed a subset of treatments for a variety of reasons
	- Example filename: CA55_g60_2_1_00-01-05.avi
		- CA55: individual ID of fly (Clogmia albipunctata number 55)
		- g60: sandpaper grit of trial (60 grit sandpaper)
		- 2: g60 was the second treatment that this fly experienced (useful to determine if performance is related to flies getting tired or damaged throughout trials)
		- 1: first video captured during this trial
		- 00-01-05 timestamp of specific walking bout
