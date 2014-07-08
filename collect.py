import pandas as pd
import numpy as np
import time
from string import lower
from stations import _get_stops, get_stop_dict, get_line, stop_for

LEX_lines = ['4','5','6']

def load_df(fname="default", setcols=['timestamp', 'id', 'stop']):
    """Load the MTA stream data into a pandas dataframe"""

    if fname=="default":
        fname = "mta_sample_stream_data.csv"

    df = pd.read_csv(fname)
    if len(setcols) != len(df.columns):
	    print "load_df Error: DataFrame has",len(df.columns),"columns"
	    print "Length does not match specified column names",setcols
	    return None
    df.columns = setcols
    # extract the train line from id
    df['line'] = map(lambda x: x.split("_")[1][0], df['id'])
    return df

def filter_stops(df, stops=["631N"], lines=['4','5']):
    """ Return entries from the supplied dataframe that match requested stops and lines

    Parameters
    required first argument - pandas DataFrame as configured by load_df
    stops - optional list of strings (default=["631N"]
    lines - optional list of strings (default=['4','5'])
    """
    filterStops = df[df['stop'].isin(stops)]
    return filterStops[filterStops['line'].isin(lines)]

def get_TOD_reference(t):
    """Return POSIX timestamp value for midnight on the day data were collected.
    This function allows referencing trip data to time of day.

    required argument is a 9-digit POSIX timestamp.
    """
    t_ref = list(time.localtime(t))
    t_ref[3:6] = [0,0,0]
    return time.mktime(t_ref)

def TOD_value(t,t_ref):
    """ Return relative time."""
    return t - t_ref

def nice_time(t, military=True):
    """Format raw time-of-day argument in seconds"""
    hh = int(t/3600.)
    mm = int((t%3600.)/60.)
    suffix = ""
    if not military:
        suffix = "AM"
        if hh>12:
            return str(hh%12) + ":" + str(mm) + "PM"
    return str(hh) + ":" + str(mm) + suffix

def df_stop_frequency(direction, for_lines=['4','5','6'], fname="default", write_df_root="stopFreq", dt=120):
        D = load_df(fname)

        get_stops = _get_stops(direction=direction, set_range="all")
        D = filter_stops(D, get_stops, for_lines)
        endpoint = "N"
        origin = "S"
        if lower(direction)[0] == "s":
                endpoint = "S"
                origin = "N"
        elif lower(direction)[0] != "n":
                print "Specified direction",direction,"not recognized; forcing NORTH"
        #station_codes = get_stop_dict(direction)
        times = D['timestamp'].unique()
	t_max = times.max()
	t_min = times.min()
	print t_min,t_max
	nbins = int((t_max - t_min)/dt) + 1
	set_index = [int(t_min+n*dt) for n in range(nbins)]
        stops = D['stop'].unique()
	print "Filtered DF contains stops",stops
	df_freq = pd.DataFrame(index=set_index, columns=stops)
	df_freq = df_freq.fillna(0.)

	for i in D.index:
		t_bin = int((D.loc[i,'timestamp'] - t_min)/dt)*dt + int(t_min)
		df_freq.loc[t_bin, D.loc[i,'stop']] = df_freq.loc[t_bin, D.loc[i,'stop']] + 1
	df_freq.to_csv(write_df_root+".csv")
	return df_freq

def df_trips_by_column(direction, for_lines=['4','5','6'], fname="default", write_df_root="tripData"):
        D = load_df(fname)
	return df_trips_by_column(D, direction=direction, for_lines=for_lines, fname=fname, write_df_root=write_df_root)

def df_trips_by_column(D, direction="N", for_lines=['4','5','6'], fname="default", write_df_root="tripData"):
        D['tref'] = D['timestamp'].map(lambda t: get_TOD_reference(t))
        # This is cumbersome and probably inefficient...
        # But it's what I do to generate unique trip_ids to manipulate
        # Probably want to end up doing this as a pre-processing step
        D['long_id'] = D['id'] + "::" + D['tref'].astype('string')

        stops = _get_stops(direction=direction, set_range="all")
        D = filter_stops(D, stops, for_lines)
        endpoint = "N"
        origin = "S"
        if lower(direction)[0] == "s":
                endpoint = "S"
                origin = "N"
        elif lower(direction)[0] != "n":
                print "Specified direction",direction,"not recognized; forcing NORTH"
        station_codes = get_stop_dict(direction)

        trips = D['long_id'].unique()
        stops = D['stop'].unique()

        tripCol = pd.DataFrame(index=['line','trip_time','tref']+list(stops), columns = trips)
        tripTimes = pd.DataFrame(index = trips, columns = ['line','trip_time','time_of_day'])
        stopCounts = pd.DataFrame(index=list(stops), columns = trips)

        ## Should profile these loops and speed it up
        for trip in trips:
                D_trip = D[D['long_id']==trip]
                l = get_line(trip)
                tripCol.loc['line',trip] = l
                tripCol.loc['tref',trip] = D_trip['tref'].values[0]
                tripTimes.loc[trip,'line'] = l
                tripTimes.loc[trip,'time_of_day'] = l
                for stop in D_trip['stop'].unique():
                        D_trip_for_stop = D_trip[D_trip['stop']==stop]
                        tripCol.loc[stop,trip] = D_trip_for_stop['timestamp'].max()
			stopCounts.loc[stop,trip] = len(D_trip_for_stop)
                trip_time = tripCol.loc[station_codes[stop_for(l,endpoint)],trip] - \
                        tripCol.loc[station_codes[stop_for(l,origin)],trip]
                tripCol.loc['trip_time',trip] = trip_time
                tripTimes.loc[trip,'trip_time'] = trip_time
                tripTimes.loc[trip,'time_of_day'] = \
                        TOD_value(tripCol.loc[station_codes[stop_for(l,origin)],\
                        trip],tripCol.loc['tref',trip])/3600.

        print "created DataFrames with data compiled by trip; writing to files with root", write_df_root
        tripCol.to_csv(write_df_root + "_verbose.csv")
        tripTimes.to_csv(write_df_root + "_trip_times.csv")
        stopCounts.to_csv(write_df_root + "_stop_counts.csv")
        return tripCol, tripTimes, stopCounts

def load_df_from_file(fname="tripData_verbose.csv"):
        tripCol = pd.read_csv(fname)
        print "loaded DF from file"
        index_colname = tripCol.columns[0]
        if index_colname != "Unnamed: 0":
                print "Expected first column named Unnamed: 0 containing index labels but got",index_colname
                print "Assigning index to values in that column. Check that this makes sense for your data."
        tripCol.index = tripCol[index_colname]
        tripCol = tripCol.drop(index_colname, axis=1)
        return tripCol

def update_depart_time(v):
	if v['depart'] > v['arrive']:
		return v['depart']
	return v['arrive']

def process(fname, file_root=None):
	tstamp = time.time()
	if not file_root:
		file_root = "subway_data_" + str(int(tstamp)) + "_"
	D = pd.read_csv(fname)
	print "Read in raw data:",np.shape(D)
	standard_cols = ['timestamp','trip_id','start_date','stop','arrive','depart']
	D.columns = standard_cols
	D['line'] = D['trip_id'].map(lambda x: x.split("_")[1][0])
	lines = D['line'].unique()
	lines = [l for l in lines if l not in [".","G"]]
	print "Collected data for subway lines",lines
        D['tref'] = D['timestamp'].map(lambda t: get_TOD_reference(t))
        D['long_id'] = D['trip_id'] + "::" + D['tref'].astype('string')

	for l in lines:
		DL = D[D['line']==l]
		all_stops = DL['stop'].unique()
		for direction in ["N","S"]:
			stops = [s for s in all_stops if s[-1]==direction]
			direction_mask = [s in stops for s in DL['stop']]
			DL_dir = DL[direction_mask]
			## pre-process
			print "Pre-processing data for line",l,"direction",direction
			#print DL_dir['stop'].unique()
			trips = DL_dir['long_id'].unique()
			times = DL_dir['timestamp'].unique()
			print "Reconciling arrive and depart data"
			DL_dir['hold'] = DL_dir['depart'] > DL_dir['arrive']
			DL_dir['hold'].value_counts()
			DL_dir['depart'] = DL_dir.apply(update_depart_time,axis=1)
			DL_dir['late'] = DL_dir['timestamp'] > (DL_dir['depart'])

			DL_dir.to_csv(file_root + l + "_" + direction + "_whole.csv")
			print "Wrote DataFrame",file_root + l + "_" + direction + "_whole.csv"
			print "Now processing data frames for line",l,direction
			## the DataFrames we'll write
			latestRecord = pd.DataFrame(index=trips, columns=stops)
			isLate = pd.DataFrame(index=trips, columns=stops)
			keepIndex = pd.DataFrame(index=trips, columns=stops)
			dropIndex = []
			######

			## very slow loop - TODO: speedup (numba, other data structures?)
			for i in DL_dir.index:
				timestamp = DL_dir.loc[i,'timestamp']
				stop = DL_dir.loc[i,'stop']
				trip_id = DL_dir.loc[i,'long_id']
				scheduled = DL_dir.loc[i,'depart']
				#print i,timestamp,stop,trip_id
				if np.isnan(latestRecord.loc[trip_id,stop]):
					latestRecord.loc[trip_id,stop] = timestamp
					keepIndex.loc[trip_id,stop] = i
					isLate.loc[trip_id,stop] = timestamp - scheduled
				else:
					old_t = latestRecord.loc[trip_id,stop]
					old_i = keepIndex.loc[trip_id,stop]
					if timestamp > old_t:
						dropIndex.append(old_i)
						latestRecord.loc[trip_id,stop] = timestamp
						keepIndex.loc[trip_id,stop] = i
						isLate.loc[trip_id,stop] = timestamp - scheduled
						# i,trip_id,stop,"replace",timestamp,">",old_t,old_i
					else:
						dropIndex.append(i)
						#print i,trip_id,stop,"REJECT",timestamp,"<",old_t,old_i
			print "assembled records to drop",len(dropIndex)
			DLdrop = DL_dir.drop(dropIndex,axis=0)
			DLdrop.to_csv(file_root + l + "_" + direction + "_clean.csv")
			print "Wrote DataFrame",file_root + l + "_" + direction + "_clean.csv"
			latestRecord.to_csv(file_root + l + "_" + direction + "_stoptimes.csv")
			print "Wrote DataFrame",file_root + l + "_" + direction + "_stoptimes.csv"
			isLate.to_csv(file_root + l + "_" + direction + "_howlate.csv")
			print "Wrote DataFrame",file_root + l + "_" + direction + "_howlate.csv"
