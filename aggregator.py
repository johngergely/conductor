import pandas as pd
import numpy as np
import time

from systemData import routeData, stationLoc

TIME_RESOLUTION_FACTOR = 1e9

def _human_time(timestamp):
    return time.localtime(timestamp)

def _make_timestamp(timestring):
    dt = pd.to_datetime(timestring)
    return dt.value/TIME_RESOLUTION_FACTOR

class aggregator():
    def __init__(self, fname):
        self.base_df = pd.read_csv(fname, index_col=0)
        for col in self.base_df.columns:
                self.base_df[col] = pd.to_datetime(self.base_df[col]*TIME_RESOLUTION_FACTOR)
        self._data_tmax = self.base_df.max().max() 
        self._data_tmin = self.base_df.min().min()
        print "Imported historical data with date range",self._data_tmin,self._data_tmax
        self.df = self.base_df

    def reset_df(self):
        self.df = self.base_df

    def select_range(self, criteria):
        print "range selection not implemented yet^^^^^^^^^^^^^^^^^^^^^"

    def process(self, listOfSegmentTuples):
        listOfStats = []
        for (origin,dest) in listOfSegmentTuples:
            diff = self.df[dest] - self.df[origin]
            diff_secs=pd.Series(diff.dropna().values.astype(float)/TIME_RESOLUTION_FACTOR)
            stats = diff_secs.describe()
            listOfStats.append(stats)
        return listOfStats
