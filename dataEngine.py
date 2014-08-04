import numpy as np
import pandas as pd

class stationLoc():
	data = pd.DataFrame( [
		["631N","Grand Central - 42 St","40.751776","-73.9768481",(3,3)],
		["635N","14 St - Union Sq","40.734673","-73.989951",(2,2)],
		["640N","Brooklyn Bridge - City Hall","40.713065","-74.004131",(1,1)],
		["NULL_STOP","NULL_STOP","40.713065","-74.004131",(0.9,1)],
		["FINAL_STOP","FINAL_STOP","40.751776","-73.9768481",(3.1,3)]
	] )
	data.columns = ['id','name','lat','lon','rel_grid']
	data = data.set_index('id')

	def __getitem__(self, (ind, col)):
		return self.data.loc[ind, col]

class routeData():
        data = pd.DataFrame( [
		["631N","631N","FINAL_STOP",1.],
		["635N","635N","631N",1.],
		["640N","640N","635N",1.],
		["NULL_STOP","NULL_STOP","640N",1.]
        ] )
        data.columns = ['id','origin','destination','travel_time']
	data = data.set_index('id')

	def __getitem__(self, (ind, col)):
		return self.data.loc[ind, col]

class systemManager():
	def __init__(self):
                self.stopSeries = pd.Series( {
                        '640N':stopObj('640N'),
			'635N':stopObj('635N'),
			'631N':stopObj('631N')
                } )

                self.routeSeries = pd.Series( {
                        'NULL_STOP':routeObj('NULL_STOP'),
                        '640N':routeObj('640N'),
			'635N':routeObj('635N'),
			'631N':routeObj('631N')
                } )

                self.activeTrains = {}

		print "Stop Data Loaded",self.stopSeries.index

        def streamUpdate(self, newDF):
                for i__ in newDF.index:
                        self._updateTrain(newDF.loc[i__, 'trip_id'],
                                         newDF.loc[i__, 'timestamp'],
                                         newDF.loc[i__,'stop'],
                                         newDF.loc[i__,'arrive'])
               
        def evolve(self, t_now, t_ref):
                for train in self.activeTrains.keys():
                        routeID = self.activeTrains[train].attrib['routeID']
                        coords, isLate = self.routeSeries[routeID].trainPosition(train, t_now)
		        self.activeTrains[train].update_position(t_now, coords, isLate)

	def drawSystem(self, timestring):
		index_list = list(self.stopSeries.index.values)
		for k in self.activeTrains.keys():
			index_list.append(k)
		scatterData = pd.DataFrame(index=[index_list], columns=['x', 'y', 'color', 'size'])
		for stop in self.stopSeries.index:
			scatterData.loc[stop,:] = self.stopSeries[stop].data().loc[stop,:]
		for train in self.activeTrains.keys():
			scatterData.loc[train,:] = self.activeTrains[train].data().loc[train,:]
		#print "DRAW compiled data"
		#print scatterData
		#self.plotInterface.plot(scatterData, timestring)
		return scatterData

	def _updateTrain(self, trip_id, timestamp, next_stop, t_arrive):
		if not self.activeTrains.get(trip_id):
        		self.activeTrains[trip_id] = trainObj(trip_id)
        		self.routeSeries["NULL_STOP"].addTrain(trip_id, timestamp, t_arrive)
			print "new train",self.activeTrains[trip_id]['id']

		require_route_update, old_route_id, new_route_id = \
                        self.activeTrains[trip_id].update_trip(self, timestamp, next_stop, t_arrive)

                if require_route_update:
                        self.routeSeries[old_route_id].clearTrain(trip_id, timestamp)
                        self.routeSeries[new_route_id].addTrain(trip_id, timestamp, t_arrive)

class plotDataObj():
	def __init__(self):
		DF = pd.DataFrame(columns = ['x', 'y', 'color', 'size'])

	def setData(self, setDF):
		self.DF = setDF

	def x(self):
		return self.DF['x']

	def y(self):
		return self.DF['y']

	def color(self):
		return self.DF['color']

	def size(self):
		return self.DF['size']

	def data(self):
		return self.DF

class vizComponent():
	def __init__(self):
		self.plotData = plotDataObj()
		self.attrib = {}
	
	def data(self):
		return self.plotData.data()

	def __getitem__(self, item):
		if not self.attrib.get(item):
			print "ERROR: vizComponent.__getitem__()"
			print "Requested attribute",item,"not found.\nDefined attributes are",
			print self.attrib.keys()
		return self.attrib[item]

	def report(self):
		print "     attributes"
		for att in self.attrib.keys():
			print "    ",att, ":", self.attrib[att],
			if not self.attrib.get(att):
				print " <-- NOT initialized"
			else:
				print " "

	def initPlotData(self, initDF):
		self.setPlotData(initDF)
	
	def setPlotData(self, setDF):
		self.plotData.setData(setDF)

class trainObj(vizComponent):
	def __init__(self, train_id):
		vizComponent.__init__(self)
		self.attrib['id'] = train_id
		self.attrib['time_of_update'] = None
                self.attrib['routeID'] = "NULL_STOP"
		self.attrib['next_stop'] = "NULL_STOP"
		self.attrib['sched_arrival'] = 0.
		self.attrib['trip_origin'] = self._calc_trip_origin()
		self.attrib['last_stop_time'] = 0.
                self.attrib['isLate'] = False

                print "create train obj route/stop",self.attrib['routeID'],self.attrib['next_stop']

	def update_trip(self, mgrObject, time_of_update, next_stop, t_arrive):
		self.attrib['time_of_update'] = time_of_update
		self.attrib['sched_arrival'] = t_arrive

                ## passed a stop; update params for next stop
                if next_stop != self.attrib['next_stop']:
                        old_route_id = self.attrib['routeID']
                        self.attrib['routeID'] = next_stop
                        self.attrib['last_stop_time'] = time_of_update
		        self.attrib['next_stop'] = next_stop
                        self.attrib['isLate'] = False

                        return True, old_route_id, next_stop
                else:
                        if time_of_update > t_arrive:
                                self.attrib['isLate'] = True
                        return False, next_stop, next_stop

        def update_position(self, timestamp, coords, isLate):
                self.attrib['isLate'] = isLate 
		quickPlotData = pd.DataFrame(index=[self['id']], columns=['x','y','color','size'], data=[[coords[0], coords[1], self._calc_color(), float(6)]])
		self.setPlotData(quickPlotData)

        def _calc_trip_origin(self):
                return int(self.attrib['id'].split("_")[0])

        def _calc_color(self):
                lateFlag = self.attrib['isLate']
                return {True:'red', False:'green'}[lateFlag]

class stopObj(vizComponent):
	def __init__(self, stop_id):
		vizComponent.__init__(self)
		self.stationLoc = stationLoc()
		self.attrib['id'] = stop_id
		self.attrib['lat'] = self.stationLoc[stop_id, 'lat']
		self.attrib['lon'] = self.stationLoc[stop_id, 'lon']
		self.attrib['grid'] = np.array(self.stationLoc[stop_id, 'rel_grid'])
		self.setPlotData(pd.DataFrame(index=[self['id']], columns=['x','y','color','size'], data=[[float(self.attrib['grid'][0]), float(self.attrib['grid'][1]), 'blue', float(10)]]))

class routeObj(vizComponent):
	def __init__(self, route_id):
		vizComponent.__init__(self)
                self.routeData = routeData()
                self.stationLoc = stationLoc()
		self.attrib['id'] = route_id
                self.attrib['origin_stop'] = self.routeData[route_id, 'origin']
                self.attrib['dest_stop'] = self.routeData[route_id, 'destination']
                self.attrib['travel_time'] = self.routeData[route_id, 'travel_time']
                
                self.origin_coord = np.array(self.stationLoc[self.attrib['origin_stop'], 'rel_grid'])
                self.dest_coord = np.array(self.stationLoc[self.attrib['dest_stop'], 'rel_grid'])
                self.activeTrains = {}

        def trainPosition(self, trip_id, timestamp):
                t_start, t_arrive, isLate = self.activeTrains[trip_id]
                progress_fraction = max(0., float(timestamp - t_start)/(t_arrive - t_start))
                if progress_fraction > 0.95:
                        progress_fraction = 0.98
                        self.activeTrains[trip_id] = t_start, t_arrive, True
                        isLate = True
		print trip_id,timestamp,t_start,t_arrive,self.origin_coord, self.dest_coord,progress_fraction
		updateCoord = self.origin_coord + progress_fraction * (self.dest_coord - self.origin_coord)
                return updateCoord, isLate

        def addTrain(self, trip_id, t_start, t_arrive):
		isLate = False
		if t_start > t_arrive:
			print trip_id, "behind schedule", t_start, t_arrive
			print "Using arbitrary trip time"
			t_arrive = t_start + 300
			isLate = True
                self.activeTrains[trip_id] = (t_start, t_arrive, isLate)

        def clearTrain(self, trip_id, timestamp):
                self.activeTrains.pop(trip_id)
	
if __name__=="__main__":
        print "dataEngine::__main__"
