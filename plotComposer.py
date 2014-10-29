import numpy as np
import pandas as pd
import time

from dataEngine import systemManager
from streamManagers import streamSimulator, liveStreamReader
from plotInterfaces import bokehPlotInterface

from collect import nice_time

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
        self.T = time.time()
        if use_T0:
                self.T = use_T0
        self.Tend = self.T + self.Tend
	T_last_update = 0.
	T_last_plot = 0.
	initPlot = False

        while self.T < self.Tend:
	    if not initPlot:
	        self.plotMgr.init_area(self.mgr.plot_boundaries())
		initPlot = True

	    if self.T - T_last_update > self.updateDataInterval:
		T_last_update = self.T
            	t_update, updateDF = self.streamMgr.read(self.T, self.T+self.updateDataInterval)
                if len(updateDF)>0:
            	        self.mgr.streamUpdate(updateDF)
                else:
                        print "Skipped processing DF of length ZERO"
	    	t_lag = time.time() - t_update
                print nice_time(t_update), nice_time(self.T), "lag between current wall-clock time and real-time feed update:","%.1f" % t_lag, "seconds"

            self.T = time.time()
            if self.T - T_last_plot > self.refreshPlotInterval:
		t_plot = t_update + self.T - T_last_update
                self.mgr.evolve(t_plot, t_update)
                stationData, trainData, routeData, fields, hoverFields = self.mgr.drawSystem(t_plot)
		scatterData = pd.concat([stationData, trainData], axis=0)
                self.plotMgr.plot(scatterData, routeData, fields, hoverFields, t_plot)
		T_last_plot = self.T

	    #time.sleep((1./self.acceleration)*self.refreshPlotInterval)
            self.T = time.time()#t_plot + self.refreshPlotInterval

if __name__ == "__main__":
    mgr = systemManager(setLines=['1','2','3','4','5','6'], setDirections=['N','S'])
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
