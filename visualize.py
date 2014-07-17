import pandas as pd
import numpy as np
from bokeh.plotting import *
from bokeh.objects import Range1d
from collect import *
from stations import *
import os

file_timestamp = "1404840164"
db = pd.DataFrame(columns=['line','direction','file','modified','first','last'],
		data=[['4','N','data/subway_data_' + file_timestamp + '_4_N','','',''],
		['4','S','data/subway_data_' + file_timestamp + '_4_S','','',''],
		['5','N','data/subway_data_' + file_timestamp + '_5_N','','',''],
		['5','S','data/subway_data_' + file_timestamp + '_5_S','','',''],
		['6','N','data/subway_data_' + file_timestamp + '_6_N','','',''],
		['6','S','data/subway_data_' + file_timestamp + '_6_S','','','']
		] )

def late_color(l,threshhold):
    if l > threshhold:
        return '#FF0000'
    else:
        return '#0066CC'

def late_alpha(l,lmax):
    if l < 0:
        return 0.1
    else:
        return min(1, 0.1+ 0.9*l/lmax)

def late_shift(x0,l,threshhold):
    if l > threshhold:
        return x0
    else:
        return -x0
    
def correct_time(t):
    if t < 0:
        return t+1440
    elif t > 1440:
        return t-1440
    else:
        return t

def jitter(x, dx):
    a0 = np.random.random()-0.5
    return x + 2*a0*dx

def plot_trip_trajectories(for_lines=['4','5','6'], for_directions=['N','S']):
	for set_line in for_lines:
		db_l = db[db['line']==set_line]
		for d in for_directions:
			db_index = db_l[db_l['direction']==d].index.values
			print "referred to db index",db_index
			file_root = db.loc[db_index,'file'].values[0]
			print file_root
			df_stoptimes = pd.read_csv(file_root+"_stoptimes.csv",index_col=0)
			df_howlate = pd.read_csv(file_root+"_howlate.csv",index_col=0)
			station_order, code_order, station_map, station_index = get_station_map(d, set_line)
			first_stop = code_order[0]
			last_stop = code_order[-1]
			df_triptimes = pd.DataFrame(index=df_stoptimes.index, columns=['trip_time'])
			df_triptimes['trip_time'] = (df_stoptimes[last_stop] - df_stoptimes[first_stop])/60.
			useColor = "#0066CC"
			output_file("plot_"+set_line+"_"+d+".html")
			figure(x_range=station_order)
			set_y_max = []
			hold()

			dropStations = [s for s in df_stoptimes.columns if s not in code_order]
			print "Dropping anomalous stations",dropStations
			print [station_names[s] for s in dropStations]
			df_stoptimes = df_stoptimes.drop(dropStations,axis=1)
			df_howlate = df_howlate.drop(dropStations,axis=1)

			lmax = df_howlate.max().max()
			print "max late",lmax
			numPlots = 0
			# generate hundreds of plots... takes a minute
			for trip in df_stoptimes.index:
				raw_record = df_stoptimes.loc[trip,:]
				#print col,raw_record
				record = pd.Series(data=[t for t in raw_record], index=df_stoptimes.columns)
		    		late = df_howlate.loc[trip,:].fillna(0)
    
    				if not np.isnan(record[first_stop]):
        				dT = (record.dropna() - record[first_stop])/60.
					if max(dT) > 0:
						set_y_max.append(max(dT))
        				useX = [1+station_map[sta] for sta in dT.index]
				        useX_shift = [late_shift(0.1,30.,l)+x for x,l in zip(useX,late)]
					
            
        				useAlpha = [late_alpha(lll,600.) for lll in late]
        				colorList = [late_color(lll,30.) for lll in late]
        				dT = [correct_time(t) for t in dT]
        
        				circle(useX_shift, y=dT, line_color=colorList, fill_color=colorList, alpha=useAlpha)

        				line(useX, y=dT, color=useColor, alpha=0.02)
        			numPlots += 1

			print "Set y max",max(set_y_max)
			if max(set_y_max)> 200:
				use_y_max = sum(set_y_max)*2.5/len(set_y_max)
			else:
				use_y_max = max(set_y_max)
			print "Use y max",use_y_max
			curplot().title = "Subway Trip Trajectories - " + str(set_line) + " Train " + str(d) + "B"
			curplot().y_range=Range1d(start=-5, end=use_y_max+5)
			xaxis().major_label_orientation = np.pi/4
			xaxis().axis_label = "Subway Stop"
			yaxis().axis_label = "Elapsed Time (min)"
			show()
			print "Plotted",numPlots,"trajectories"


def plot_departure_delays(for_lines=['4','5','6'], for_directions=['N','S']):
	for l in for_lines:
		db_l = db[db['line']==l]
		for d in for_directions:
			db_index = db_l[db_l['direction']==d].index.values
			print "referred to db index",db_index
			file_root = db.loc[db_index,'file'].values[0]
			print file_root
			#df_stoptimes = pd.read_csv(file_root+"_stoptimes.csv",index_col=0)
			df_howlate = pd.read_csv(file_root+"_howlate.csv",index_col=0)
			station_order, code_order, station_map, station_index = get_station_map(d, l)
			useColor = "#0066CC"
			output_file("plot_delays_"+l+"_"+d+".html")
			figure(x_range=station_order)
			hold()

			dropStations = [s for s in df_howlate.columns if s not in code_order[1:-1]]
			print "Dropping anomalous stations",dropStations
			print [station_names[s] for s in dropStations]
			#df_stoptimes = df_stoptimes.drop(dropStations,axis=1)
			df_howlate = df_howlate.drop(dropStations,axis=1)

			useY = []
			useX = []
			useS = []
			for stop in df_howlate.columns:
				useX.append(1+station_map[stop])
    				useY.append(max(5,min(60,df_howlate[stop].mean())))
    				#useY.append(max(5,int(df_howlate[df_howlate[stop]>20.][stop].mean()/30.)*5))
    				s_total = df_howlate[stop].count()
    				s_late = sum(df_howlate[stop]>30.)
    				useS.append(float(s_late)/s_total * 100)
    
    				scatter(useX, y=useS, alpha=0.4, color=useColor, size=useY)

			#curplot().y_range = Range1d(start=-5, end=60)
			curplot().title = "Departure Delays by Station - " + l + " Train " + d + "B"
			xaxis().major_label_orientation = np.pi/4
			xaxis().axis_label = "Subway Stop"
			yaxis().axis_label = "Percentage of Late Departures"
			show()
			agg_late = sum([0.01*p*t for p,t in zip(useS, useY)])
			print l,d,agg_late


import re

def retrieve_schedules(DOW, line, direction,huge):
    DAY = "WKD"
    if DOW==5:
        DAY = "SAT"
    elif DOW==6:
        DAY = "SUN"
    depart_wildcard = "[0-9][0-9][0-9][0-9][0-9][0-9]"
    trip_regex = DAY + "_" + depart_wildcard + "_" + line + "_" + direction
    matches = re.findall(trip_regex, huge)
    if len(matches)==0:
        print "match NOT found for",trip_regex
        return None
    #trip_string = DAY + "_" + depart_time + "_" + line + "_" + direction
    return matches

def to_hhmmss(t_0):
	t_mins = t_0/100
	return (str(t_mins/60) + ":" + str(t_mins%60) + ":" + str(t_0%100))

def plot_time_evolution(for_lines=['4','5','6'], for_directions=['N','S']):
	for l in for_lines:
		db_l = db[db['line']==l]
		for d in for_directions:
			db_index = db_l[db_l['direction']==d].index.values
			print "referred to db index",db_index
			file_root = db.loc[db_index,'file'].values[0]
			print file_root
			D_stoptimes = pd.read_csv(file_root+"_stoptimes.csv",index_col=0)
			#df_howlate = pd.read_csv(file_root+"_howlate.csv",index_col=0)
			station_order, code_order, station_map, station_index = get_station_map(d, l)
			useColor = "#0066CC"
			output_file("plot_time_evol_"+l+"_"+d+".html")

			keepStations = [sta for sta in code_order]
			dropStations = [sta for sta in D_stoptimes.columns if sta not in code_order]
			print "dropping", dropStations
			D_stoptimes = D_stoptimes.drop(dropStations, axis=1)
			#tmpD = D_stoptimes * 1e9
			#D_datetime = tmpD.astype('M8[ns]')

			print "Initializing figure"
			figure(y_axis_type='datetime')
			hold()
			print "Entering first loop"
			for trip in D_stoptimes.index:
		    		raw_data = D_stoptimes.loc[trip,:].dropna()
				tmpD = raw_data * 1e9
				data_datetime = tmpD.astype('M8[ns]')
    				x_data = [station_map[data_datetime.index[i]] for i in range(len(data_datetime))]
    				y_data = [data_datetime[i].tz_localize('US/Eastern') for i in data_datetime.index]

    				scatter(x_data, y=y_data, alpha=0.15, size=5, color=useColor)
    				line(x_data, y_data, alpha=0.05, color=useColor)

			print "Through first loop"
			#sched = pd.read_csv("data/schedule_hhmmss.csv", index_col=0)
			sched = pd.read_csv("data/schedule.csv", index_col=0)
			sched = sched[keepStations]
			sched = sched.dropna(axis=0, how='all')
			print "trimmed schedule df to",np.shape(sched)
			huge = ''.join(sched.index.values)

			T0 = D_stoptimes.min().min()
			T0 = pd.to_datetime(T0*1e9, 'M8[ns]')
			Tmax = D_stoptimes.max().max()
			Tmax = pd.to_datetime(Tmax*1e9, 'M8[ns]')
			datesToPlot = pd.date_range(start=T0, end=Tmax, freq='D')
			#datesToPlot.tz_localize('US/Eastern')

			useIndex = D_stoptimes.columns
			x_data = [station_map[d_index] for d_index in useIndex]
			DOW = None
			for plotDate in datesToPlot:
    				if plotDate.dayofweek != DOW:
        				DOW = plotDate.dayofweek
        				sched_strings = retrieve_schedules(DOW, l, d, huge)
    				base_date_str = str(plotDate.date())
   
   				print "plotting schedule data for date",plotDate.date()
    				for sched_index in sched_strings:
        				sched_data = sched.loc[sched_index,:].dropna()
        				y_data = [pd.to_datetime(base_date_str + " " + 
            					to_hhmmss(sched_data[sta]))
						for sta in useIndex if sta in sched_data.index]
        				#y_data = [yy.tz_localize('US/Eastern') for yy in y_data]
        
        				line(x_data, y_data, alpha=0.15, color='red')

			curplot().title = "Time Evolution of Subway Stops - " + l + " Train " + d + "B"
			xaxis().major_label_orientation = np.pi/4
			xaxis().axis_label = "Subway Stop"
			yaxis().axis_label = "Date & Time"
			show()


def plot_trip_intervals(for_lines=['4','5','6'], for_directions=['N','S']):
	for set_line in for_lines:
		db_l = db[db['line']==set_line]
		for d in for_directions:
			db_index = db_l[db_l['direction']==d].index.values
			print "referred to db index",db_index
			file_root = db.loc[db_index,'file'].values[0]
			print file_root
			#D_stoptimes = pd.read_csv(file_root+"_stoptimes.csv",index_col=0)
			D_howlate = pd.read_csv(file_root+"_howlate.csv",index_col=0)
			station_order, code_order, station_map, station_index = get_station_map(d, set_line)
			useColor = "#0066CC"
			output_file("plot_trip_intervals_"+set_line+"_"+d+".html")

			keepStations = [sta for sta in code_order]
			dropStations = [sta for sta in D_howlate.columns if sta not in code_order]
			print "dropping", dropStations
			D_howlate = D_howlate.drop(dropStations, axis=1)
			t_range = 5

			figure(x_range=station_order)#, y_range=Range1d(start=-1, end=t_range))
			hold()

			for trip in D_howlate.index:
    				raw_record = D_howlate.loc[trip,:]
    				ordered_record = np.array([raw_record[sta_code]/60. for sta_code in code_order])
    				record = ordered_record[1:-1] - ordered_record[0:-2]
    				record_jitter = [jitter(r,0.5) for r in record]
    				dT = pd.Series(data=record_jitter, index=code_order[1:-1])
    				dT = dT.dropna()
    				late = D_howlate.loc[trip,:]
    				useX = [late_shift(0.1,l,30.)+1+station_map[sta] for sta,l in zip(dT.index,late)]
    				colorList = [late_color(ll,30.) for ll in late]
    				#alphaList = [late_alpha(l,600.) for l in late]

    				scatter(useX, y=dT, alpha=0.15, color=colorList)

			curplot().y_range = Range1d(start=0, end=10)
			curplot().title = "Trip Time Intervals - " + str(set_line) + " Train " + str(d) + "B"
			xaxis().major_label_orientation = np.pi/4
			xaxis().axis_label = "Approaching Subway Stop"
			yaxis().axis_label = "Trip Time (minutes)"
			show()









#### this should all be removed once dependencies are eliminated...
def plot_init():
	# set this if you want
	overwrite_files = False
	if os.path.exists("tripData_verbose.csv") and not overwrite_files:
		print "Load from existing files..."
	    	tCol = load_df_trips_by_column("tripData_verbose.csv")
		tTrip = load_df_trips_by_column("tripData_trip_times.csv")
    	else:
		tCol, tTrip = df_trips_by_column("N")

	output_file("bokeh_plot.html")

	tTrip = tTrip.dropna()
	## cut out unreasonable times that are artifacts of
	## midnight crossings (these can be dealt with) and glitches in data collection
	print len(tTrip),"original records"
	dt_max = 150*60.
	dt_min = 45*60.
	df_mask = (tTrip['trip_time']>dt_min) & (tTrip['trip_time']<dt_max)
	tTrip = tTrip[df_mask]
	print len(tTrip),"records after culling bad data"
	xval = tTrip['time_of_day'].values
	yval = tTrip['trip_time'].values/60.
	colors = [line_color(str(l)) for l in tTrip['line'].values]
	print tCol.index
	print tCol.shape
	bad_trips = tCol.loc['trip_time'].isnull()
	print len(tCol.columns[bad_trips])
	tCol = tCol.drop(tCol.columns[bad_trips], axis=1)
	cull_cols = (tCol.loc['trip_time']<dt_min) | (tCol.loc['trip_time']>dt_max)
	print len(tCol.columns[cull_cols])
	tCol = tCol.drop(tCol.columns[cull_cols], axis=1)
	tCol_stops = tCol.drop(['line','trip_time','tref'],axis=0)
	#print tCol_stops
	print tCol_stops.shape
	return tCol_stops, tCol

def route_trace_plot(tCol_stops, tCol):
	station_codes = get_stop_dict("N")
	code_order = [station_codes[s] for s in station_order]
	print code_order
	station_map = dict(zip(code_order, np.arange(len(code_order))))
	station_index = dict(zip(np.arange(len(code_order)), code_order))
	print station_map
	ref_stop = station_codes['Brooklyn_Bridge']
	print ref_stop
	figure(y_range=station_order)
	hold()
	Y = [1+station_map[sta] for sta in tCol_stops.index.values]
	print Y
	# generate 1K plots... takes a minute
	for col in tCol_stops.columns:
	    record = tCol_stops[col].astype(float)
	    #if str(int(tCol.loc['line',col])) == '6':
	    if not np.isnan(record[ref_stop]):
        	dT = (record.dropna() - record[ref_stop])/60.
	        useY = [1+station_map[sta] for sta in dT.index]
	        #print col
        	#print dT
	        #print useY
        
        	line(x=dT, y=useY, 
	             color=line_color(str(int(tCol.loc['line',col]))), 
        	     alpha=0.1)

	snippet = curplot().create_html_snippet(embed_base_url='../static/js/', embed_save_loc='./static/js')
	#show()
	return snippet

if __name__=="__main__":
	#plot_time_evolution()
	#plot_departure_delays()
	#plot_trip_intervals()
	plot_trip_trajectories()
