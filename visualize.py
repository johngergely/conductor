import pandas as pd
import numpy as np
from bokeh.plotting import *
from bokeh.objects import Range1d
from collect import *
from stations import *
import os

db = pd.DataFrame(columns=['line','direction','file','modified','first','last'],
		data=[['4','N','subway_data_1404770776_4_N','','',''],
		['4','S','subway_data_1404840164_4_S','','',''],
		['5','N','subway_data_1404840164_5_N','','',''],
		['5','S','subway_data_1404840164_5_S','','',''],
		['6','N','subway_data_1404840164_6_N','','',''],
		['6','S','subway_data_1404840164_6_S','','','']
		] )

def late_color(l,threshhold):
    if l > threshhold:
        return '#FF0000'
    else:
        return '#0066CC'

def late_alpha(l,lmax):
    return min(1, 0.1+ 0.9*l/lmax)
    
def correct_time(t):
    if t < 0:
        return t+1440
    return t

def plot_trip_trajectories(for_lines=['4','5','6'], for_directions=['N','S']):
	for l in for_lines:
		db_l = db[db['line']==l]
		for d in for_directions:
			db_index = db_l[db_l['direction']==d].index.values
			print "referred to db index",db_index
			file_root = db.loc[db_index,'file'].values[0]
			print file_root
			df_stoptimes = pd.read_csv(file_root+"_stoptimes.csv",index_col=0)
			df_howlate = pd.read_csv(file_root+"_howlate.csv",index_col=0)
			station_order, code_order, station_map, station_index = get_station_map(d, l)
			first_stop = code_order[0]
			last_stop = code_order[-1]
			df_triptimes = pd.DataFrame(index=df_stoptimes.index, columns=['trip_time'])
			df_triptimes['trip_time'] = (df_stoptimes[last_stop] - df_stoptimes[first_stop])/60.
			useColor = "#0066CC"
			output_file("plot_"+l+"_"+d+".html")
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
            
        				useAlpha = [late_alpha(lll,600.) for lll in late]
        				colorList = [late_color(lll,30.) for lll in late]
        				dT = [correct_time(t) for t in dT]
        
        				circle(useX, y=dT, line_color=colorList, fill_color=colorList, alpha=useAlpha)

        				line(useX, y=dT, color=useColor, alpha=0.02)
        			numPlots += 1

			print "Set y max",max(set_y_max)
			if max(set_y_max)> 200:
				use_y_max = sum(set_y_max)*2.5/len(set_y_max)
			else:
				use_y_max = max(set_y_max)
			print "Use y max",use_y_max
			curplot().title = "Subway Trip Trajectories - " + str(l) + " Train " + str(d) + "B"
			curplot().y_range=Range1d(start=-5, end=use_y_max+5)
			xaxis().major_label_orientation = np.pi/4
			xaxis().axis_label = "Subway Stop"
			yaxis().axis_label = "Elapsed Time (min)"
			show()
			print "Plotted",numPlots,"trajectories"




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
