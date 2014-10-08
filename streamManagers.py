import pandas as pd
import numpy as np
from read_stream import readMTADataStream

class streamManager():
    def __init__(self):
        pass

    def read(self):
        print "streamManager.read() must be implemented by derived class!",self

class liveStreamReader(streamManager):
    def __init__(self):
        streamManager.__init__(self)
        self.T0 = None

    def read(self, t1, t2):
        nyct_feed = None
        attempts = 0
        while not nyct_feed:
            attempts += 1
            nyct_feed = readMTADataStream()
        timestamp = nyct_feed.header.timestamp
        self.T0 = timestamp
        count = 0
        newDF = pd.DataFrame()
        for entity in nyct_feed.entity:
                if entity.trip_update.trip.trip_id:
                        stops = [stu for stu in entity.trip_update.stop_time_update]
                        if len(stops)>0:
                                count += 1
                                newDF = newDF.append([[
                                        int(timestamp),
                                        str(entity.trip_update.trip.trip_id),
                                        str(entity.trip_update.trip.start_date),
                                        str(stops[0].stop_id),
                                        float(stops[0].arrival.time),
                                        float(stops[0].departure.time)
                                ]])
        print "liveStreamReader ==> Read", len(newDF), "records after",attempts,"attempts."
        if len(newDF) == 0:
            print "returning length ZERO dataframe from stream read"
        else:
            newDF.columns = ['timestamp','trip_id','start_date','stop','arrive','depart']
            newDF.index = np.arange(len(newDF))
        return self.T0, newDF

class streamSimulator(streamManager):
    def __init__(self, fname):
        streamManager.__init__(self)
        self.df = pd.read_csv(fname)
        self.initDF()
        ## expect timestamps in this df to be ordered (but this is not checked)

    def initDF(self):
        times = self.df['timestamp'].unique()
        self.T0 = self.df['timestamp'].min()
        self.Tfinal = self.df['timestamp'].max()
        print "Datafame contains range of timestamps",self.T0,self.Tfinal
        self.index_position = 0
	self.index_max = len(self.df) - 1

    def read(self, t1, t2):
	index_range = []
        t__ = self.df.loc[self.index_position, 'timestamp']
        while t__ < t1:
            self.index_position = self.index_position + 1
            t__ = self.df.loc[self.index_position, 'timestamp']

        while t__ <= t2 and self.index_position<self.index_max:
	    index_range.append(self.index_position)
            self.index_position = self.index_position + 1
            t__ = self.df.loc[self.index_position, 'timestamp']
	#print "    collect range",index_range

        return t1, self.df.loc[index_range,:]
