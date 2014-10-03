import pandas as pd
import numpy as np
import time

from systemData import routeData, stationLoc
from visualize import file_timestamp

TIME_RESOLUTION_FACTOR = 1e9

NULL_STATS = {"min":240, "50%":240, "75%":240}

def _human_time(timestamp):
    return time.localtime(timestamp)

def _make_timestamp(timestring):
    dt = pd.to_datetime(timestring)
    return dt.value/TIME_RESOLUTION_FACTOR

class aggregator():
    def __init__(self, ll, dd):
        ## needs to be more robust access to database
	fname_string = "data/subway_data_" + file_timestamp + "_" + ll + "_" + dd + "_stoptimes.csv"
        self.base_df = pd.read_csv(fname_string, index_col=0)
        for col in self.base_df.columns:
                self.base_df[col] = pd.to_datetime(self.base_df[col]*TIME_RESOLUTION_FACTOR)
        self._data_tmax = self.base_df.max().max() 
        self._data_tmin = self.base_df.min().min()
        print "Imported historical data with date range",self._data_tmin,self._data_tmax
        self.df = self.base_df

    def reset(self):
        self.df = self.base_df

    def select_range(self, criteria):
        print "range selection not implemented yet^^^^^^^^^^^^^^^^^^^^^"

    def process(self, listOfSegmentTuples):
        listOfStats = []
        for (origin,dest) in listOfSegmentTuples:
	    listOfStats.append(self.process_tuple(self, origin, dest))
	return listOfStats

    def process_tuple(self, origin, dest):
	if origin not in self.df.columns:
	    return NULL_STATS
	if dest not in self.df.columns:
	    return NULL_STATS
        diff = self.df[dest] - self.df[origin]
        diff_secs=pd.Series(diff.dropna().values.astype(float)/TIME_RESOLUTION_FACTOR)
        stats = diff_secs.describe()
        return stats

########
########

class intervalAggregator():
        def __init__(self, ll, dd):
                ## needs to be more robust access to database
                fname_string = "data/subway_data_" + file_timestamp + "_" + ll + "_" + dd + "_stoptimes.csv"
                self.df = pd.read_csv(fname_string, index_col=0)
                self._data_tmax = self.df.max().max() 
                self._data_tmin = self.df.min().min()
                self.df["date"] = self.df.index.map(lambda x: x.split("::")[1])
                self.dates = self.df["date"].unique()
                self.df["time"] = self.df.index.map(lambda x: int(x.split("_")[0])/6000.)

        def calcInterval(self, hmin, hmax, station):
                df_select = self.df[self.df["time"]>hmin]
                df_select = df_select[df_select["time"]<hmax]
                if station not in df_select.columns:
                        #print "NO DATA FOR STATION",station
                        return 0.
                df_station = df_select[[station,"date"]]
                all_intervals = np.array(())
                for d in self.dates:
                        df_sub = df_station[df_station["date"]==d]
                        ticks = df_sub[station].dropna().values
                        ticks.sort
                        intervals = ticks[1:] - ticks[:-1]
                        all_intervals = np.hstack((all_intervals, intervals))
                station_mean = all_intervals.mean()/60.
                return station_mean

        def calcSeries(self,station):
                means = {}
                for hmin in range(0,24):
                        hmax = hmin+1
                        means[hmin] = self.calcInterval(hmin, hmax, station)
                return means
