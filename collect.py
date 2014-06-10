import pandas as pd
import numpy as np
import time
from string import lower

DEFINE_STOPS_LEX_AVE_NORTHBOUND = {'Utica':'250N', 'Flatbush':'247N', 'Atlantic':'235N', 'Fulton':'418N', 'Brooklyn_Bridge':'640N', 'Union':'635N', 'Grand_Central':'631N', '125_St':'621N', 'Woodlawn':'401N', 'Dyre':'501N', 'Pelham':'601N'}
DEFINE_STOPS_LEX_AVE_SOUTHBOUND = {'Utica':'250S', 'Flatbush':'247S', 'Atlantic':'235S', 'Fulton':'418S', 'Brooklyn_Bridge':'640S', 'Union':'635S', 'Grand_Central':'631S', '125_St':'621S', 'Woodlawn':'401S', 'Dyre':'501S', 'Pelham':'601S'}
LEX_lines = ['4','5','6']

def load_df(fname="default"):
    """Load the MTA stream data into a pandas dataframe"""

    if fname=="default":
        fname = "mta_sample_stream_data.csv"

    df = pd.read_csv(fname)
    df.columns = ['timestamp', 'id', 'stop']
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

def _get_stops(direction, set_range="ALL", route="LEX"):
        if lower(route) != "lex":
                print "Right now only Lexington Avenue 4/5/6 routes are supported. You requested unsupported route parameter",route
                return None
        dir_flag = lower(direction[0]) 
        use_dict = DEFINE_STOPS_LEX_AVE_NORTHBOUND
        start_list = ['Utica', 'Flatbush', 'Brooklyn_Bridge']
        end_list = ['Woodlawn', 'Dyre', 'Pelham']
        others_list = [k for k in DEFINE_STOPS_LEX_AVE_NORTHBOUND.keys() if (k not in start_list and k not in end_list)]
        if dir_flag == "s":
                use_dict = DEFINE_STOPS_LEX_AVE_SOUTHBOUND
                start_list = ['Woodlawn', 'Dyre', 'Pelham']
                end_list = ['Utica', 'Flatbush', 'Brooklyn_Bridge']
        elif dir_flag != "n":
                print "Failed to return list of stops! Allowed direction parameters are [N]orthbound or [S]outhbound; You specified",direction
                return None
        if lower(set_range) == "all":
                return use_dict.values()
        elif lower(set_range) == "interim":
                return [use_dict.get(k) for k in others_list]
        elif lower(set_range) == "start":
                return [use_dict.get(k) for k in start_list]
        elif lower(set_range) == "end":
                return [use_dict.get(k) for k in end_list]
        else:
                allowed_range = ["all", "start", "end", "interim"]
                print "Failed to return list of stops! Allowed range parameters are",allowed_range,"; You specified",set_range
                return None

def get_stop_dict(direction, route="LEX"):
        """ Return the dictionary of station stops and stop codes."""
        if lower(route) != "lex":
                print "Right now only Lexington Avenue 4/5/6 routes are supported. You requested unsupported route parameter",route
                return None
        dir_flag = lower(direction[0]) 
        use_dict = DEFINE_STOPS_LEX_AVE_NORTHBOUND
        if dir_flag == "s":
                use_dict = DEFINE_STOPS_LEX_AVE_SOUTHBOUND
        elif dir_flag != "n":
                print "Failed to return list of stops! Allowed direction parameters are [N]orthbound or [S]outhbound; You specified",direction
                return None
        return use_dict

def get_line(trip_id):
    return trip_id.split("_")[1][0]

def stop_for(l,terminus):
    return {"N":{'4':'Woodlawn', '5':'Dyre', '6':'Pelham'}, "S":{'4':'Utica', '5':'Flatbush','6':'Brooklyn_Bridge'}}[terminus][l]

def df_trips_by_column(direction, for_lines=['4','5','6'], fname="default", write_df_root="tripData"):
        D = load_df(fname)
        D['tref'] = D['timestamp'].map(lambda t: get_TOD_reference(t))
        # This is probably inefficient... might want to do this as a pre-process step during data collection
        D['long_id'] = D['id'] + "::" + D['tref'].astype('string')

        stops = _get_stops(direction=direction, set_range="all")
        D = filter_stops(D, stops, for_lines)
        endpoint = "N"
        origin = "S"
        if lower(direction)[0] == "s":
                endpoint = "S"
                origin = "N"
        station_codes = get_stop_dict(direction)

        trips = D['long_id'].unique()
        stops = D['stop'].unique()

        tripCol = pd.DataFrame(index=['line','trip_time','tref']+list(stops), columns = trips)
        tripTimes = pd.DataFrame(index = trips, columns = ['line','trip_time','time_of_day'])

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
        return tripCol, tripTimes

def load_df_trips_by_column(fname="tripData_verbose.csv"):
        tripCol = pd.read_csv(fname)
        print "loaded DF from file"
        index_colname = tripCol.columns[0]
        if index_colname != "Unnamed: 0":
                print "Expected first column named Unnamed: 0 containing index labels but got",index_colname
                print "Assigning index to values in that column. Check that this makes sense for your data."
        tripCol.index = tripCol[index_colname]
        tripCol = tripCol.drop(index_colname, axis=1)
        return tripCol
