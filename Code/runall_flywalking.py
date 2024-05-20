#!/usr/bin/python3

###########################################################################
from colorama import Fore, Back, Style
###########################################################################

srcfolder = "/home/eebrandt/projects/UChicago/fly_walking/sandpaper/src"
print(Fore.BLUE +"Attempting write_final_csvs.py")
try:
	exec(open("write_final_csvs.py").read())
	print(Fore.CYAN + Style.BRIGHT + "write_final_csvs.py was successful")
except:
	print(Fore. RED + "write_final_csvs.py failed")
	sys.exit(1)
	
print(Fore.BLUE +"Attempting comparative_kinematicparams.py")	
try:
	exec(open("comparative_kinematicparams.py").read())
	print(Fore.CYAN + Style.BRIGHT + "comparative_kinematicsparams.py was successful")
	
except:
	print(Fore.RED + "comparative_kinematicsparams.py failed")
	sys.exit(1)
	
print(Fore.BLUE +"Attempting speed_between_conditions.py")		
try:
	exec(open("speed_between_conditions.py").read())
	print(Fore.YELLOW + "speed_between_conditions.py was successful")
except:
	print(Fore.CYAN + Style.BRIGHT + "speed_between_conditions.py failed")
	sys.exit(1)
	
print(Fore.BLUE +"Attempting phi_phi_diagram.py")		
try:
	exec(open("phi_phi_diagram.py").read())
	print(Fore.CYAN + Style.BRIGHT + "phi_phi_diagram.py was successful")
except:
	print(Fore.RED + "phi_phi_diagram.py failed")
	sys.exit(1)
	
print(Fore.BLUE +"Attempting footprint_analysis_plots.py")		
try:
	exec(open("footprint_analysis_plots.py").read())
	print(Fore.CYAN + Style.BRIGHT + "footprint_analysis_plots.py was successful")
except:
	print(Fore.RED + "footprint_analysis_plots.py failed")
	sys.exit(1)
	
print(Fore.BLUE +"Attempting average_gait_diagram.py")		
try:
	exec(open("average_gait_diagram.py").read())
	print(Fore.CYAN + Style.BRIGHT + "average_gait_diagram.py was successful")
except:
	print(Fore.RED + "average_gait_diagram.py failed")	
	sys.exit(1)
	
print("All done!")
	
	

