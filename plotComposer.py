import numpy as np
import pandas as pd
import time

from dataEngine import systemManager
from streamManagers import streamSimulator, liveStreamReader
from plotInterfaces import bokehPlotInterface

class plotManager():
    def __init__(self, systemManager, streamManager, plotDevice):

        self.mgr = systemManager
        self.streamMgr = streamManager
        self.plotMgr = plotDevice

        self.updateDataInterval = 30.
        self.refreshPlotInterval = 5.
        self.acceleration = 1.
        self.Tend = 3600.*24

    def setLifetime(self, Tend):
        self.Tend = Tend
        print "plotManager::::::::::(times in secs)"
        print "Trip data time span",self.Tend
        print "plot refresh interval",self.refreshPlotInterval
        print "data update interval",self.updateDataInterval

    def run(self, use_T0=None):
        if use_T0:
                self.clock = use_T0
        else: 
                self.clock = time.time()
        self.Tend = self.clock + self.Tend

        while self.clock < self.Tend:
            ## this is awkward, specifying interval will need to be redone with live stream
            updateDF = self.streamMgr.read(self.clock, self.clock+self.updateDataInterval)
            print self.clock, "[",self.clock,self.Tend,"] update DF",updateDF['timestamp'][0]
            #print updateDF[['timestamp', 'trip_id','stop','arrive']]
            self.mgr.streamUpdate(updateDF)

	    self.plotMgr.init_area(self.mgr.plot_boundaries())

            t_plot = self.clock
            t_from_last_update = self.clock
            while (t_plot < (self.clock + self.updateDataInterval)):
                self.mgr.evolve(t_plot, t_from_last_update)
                staticData, dynamicData, fields, hoverFields = self.mgr.drawSystem(t_plot)
                self.plotMgr.plot(staticData, dynamicData, fields, hoverFields, t_plot)

                t_plot = t_plot + self.refreshPlotInterval
		time.sleep((1./self.acceleration)*self.refreshPlotInterval)

            self.clock = self.clock + self.updateDataInterval
            #print "completed plot cycle",self.clock,t_plot

if __name__ == "__main__":
    mgr = systemManager(setLines=['4','5','6'], setDirections=['N','S'])
    liveStream = liveStreamReader()      
    plotDevice= bokehPlotInterface()

    boss = plotManager(mgr, liveStream, plotDevice)
    boss.run()

    ## simulation
    #mgr = systemManager()
    #stream = streamSimulator("trip_data_test.csv")
    #plotDevice= bokehPlotInterface()
    #T0, Tfinal = stream.T0, stream.Tfinal

    #boss = plotManager(mgr, stream, plotDevice)
    #boss.setLifetime(Tfinal)
    #boss.run(use_T0=T0)
