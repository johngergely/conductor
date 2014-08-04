import pandas as pd

class streamManager():
    def __init__(self):
        pass

    def read(self):
        print "streamManager.read() must be implemented by derived class!",self

class liveStreamReader(streamManager):
    def __init__(self):
        streamManager.__init__(self)

    def read(self):
        print "Need to set up pull from live stream here based on code in read_stream.py"

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

        return self.df.loc[index_range,:]
