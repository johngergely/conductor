import pandas as pd
import numpy as np
import time
import pickle

from systemData import routeData, stationLoc
from visualize import file_timestamp

from stations import station_names

TIME_RESOLUTION_FACTOR = 1e9

NULL_STATS = {}
for i in range(24):
        NULL_STATS[i] = {"min":240, "50%":240, "75%":240}

def _human_time(timestamp):
    return time.localtime(timestamp)

def _make_timestamp(timestring):
    dt = pd.to_datetime(timestring)
    return dt.value/TIME_RESOLUTION_FACTOR

## base class
## derived classes are expected to implement CalcInterval
class aggregator():
        def __init__(self, ll, dd, useComputed=True):
                self.line = ll
                self.direction = dd
                self.useComputed = useComputed
                self.firstCalc = True
                if not self.useComputed:
                        print "useComputed flag is set to false.\nThis forces re-calculation of interval data and is a time-consuming call.\nThis call should only be made occasionally when re-computes are required. Otherwise re-use archived data if possible by setting useCompute=True."

        def calcInit(self):
                ## needs to be more robust access to database
                fname_string = "data/subway_data_" + file_timestamp + "_" + self.line + "_" + self.direction + "_stoptimes.csv"
                self.df = pd.read_csv(fname_string, index_col=0)
                self._data_tmax = self.df.max().max() 
                self._data_tmin = self.df.min().min()
                self.df["date"] = self.df.index.map(lambda x: x.split("::")[1])
                self.dates = self.df["date"].unique()
                # quotient of 6000 converts from hundredths of a minute to hours 
                self.df["time"] = self.df.index.map(lambda x: int(x.split("_")[0])/6000.)
                print self.ID,"init"
                print self.df.columns

        def loadData(self):
                try:
                        with open(self.db_filename,'r') as f:
                                self.computedSeries = pickle.load(f)
                                print self.ID, "Loaded archive with",len(self.computedSeries.keys()),"records"
                except:
                        self.computedSeries = {}
                        print self.ID, "failed to load data. starting with empty dict"

        def storeData(self):
                if self.firstCalc:
                        print self.ID,"storeData called but no new data computed; nothing to store"
                else:
                        with open(self.db_filename,'w') as f:
                                pickle.dump(self.computedSeries, f)
                        print self.ID,"wrote",len(self.computedSeries),"computed station records to archive"

        def calcSeries(self, token):
                if self.firstCalc:
                        self.calcInit()
                        self.firstCalc = False
                stats_per_hour = {}
                for hmin in range(0,24):
                        hmax = hmin+1
                        stats_per_hour[hmin] = self.calcInterval(hmin, hmax, token)
                self.computedSeries[token] = stats_per_hour 
                self.storeData()
                return stats_per_hour

        def fetchSeries(self, token):
                if self.useComputed and self.computedSeries.get(token):
                        return self.computedSeries[token]
                else:
                        print self.ID," did not find records for",token
                        return self.calcSeries(token)

class segmentAggregator(aggregator):
        def __init__(self, ll, dd, useComputed=True):
                aggregator.__init__(self, ll, dd, useComputed)
                self.ID = "segmentAggregatorArchive_" + self.line + "_" + self.direction
                self.db_filename = "data/." + self.ID + ".dict"
                self.loadData()

        def calcInterval(self, hmin, hmax, (origin, dest)):
                df_select = self.df[self.df["time"]>hmin]
                df_select = df_select[df_select["time"]<hmax]
	        if origin not in df_select.columns:
	                return NULL_STATS
	        if dest not in df_select.columns:
	                return NULL_STATS
                diff = df_select[dest] - df_select[origin]
                # just drop negatives but should do something more sophisticated
                todrop = diff[diff<0].index
                diff = diff.drop(todrop)
                if len(diff[diff<0]) > 0:
                        print "NEGATIVES",origin,dest,diff[diff<0].values
                #diff_secs=pd.Series(diff.dropna().values.astype(float)/TIME_RESOLUTION_FACTOR)
                diff_secs=pd.Series(diff.dropna().values.astype(float))
                stats = diff_secs.describe()
                return stats


class frequencyAggregator(aggregator):
        def __init__(self, ll, dd, useComputed=True):
                aggregator.__init__(self, ll, dd, useComputed)
                self.ID = "intervalAggregatorArchive_" + self.line + "_" + self.direction
                self.db_filename = "data/." + self.ID + ".dict"
                self.loadData()

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
                if np.isnan(station_mean):
                        return 0.
                return station_mean

class durationAggregator(aggregator):
        def __init__(self, ll, dd, useComputed=True):
                aggregator.__init__(self, ll, dd, useComputed)
                self.ID = "durationAggregatorArchive_" + self.line + "_" + self.direction
                self.db_filename = "data/." + self.ID + ".dict"
                self.loadData()

        def calcInit(self):
            aggregator.calcInit(self)
            ## the following line of code is obscure: i am taking the minimum timestamp from the station columns, i.e. the minimum of each row in the dataframe, excluding the time and date columns
            ## i guess it's an idiosyncrasy in pandas that axis=1 as an optional argument to drop specifies **column** whereas axis=1 as an optional argument to min gives me the minimum **by row**
            self.df['min'] = self.df.drop(['date','time'],axis=1).min(axis=1)
            print self.ID, "local calcInit computed column with earliest stop time for each row"

        def calcInterval(self, hmin, hmax, station):
                df_select = self.df[self.df["time"]>hmin]
                df_select = df_select[df_select["time"]<hmax]
	        if station not in df_select.columns:
                        return 0.
                diff = df_select[station] - df_select['min']
                # just drop negatives but should do something more sophisticated
                todrop = diff[diff<0].index
                if len(diff[diff<0]) > 0:
                        print self.ID,"NEGATIVES",origin,dest,diff[diff<0].values
                diff = diff.drop(todrop)
                print self.ID,station,station_names[station],diff.mean()/60.
                return diff.mean()/60.
