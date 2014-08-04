import numpy as np
import pandas as pd

class stationLoc():
        data = pd.read_csv("stops_formatted.txt", index_col=0)
	#data.columns = ['name','lat','lon']
	#data = data.set_index('id')

	def __getitem__(self, (ind, col)):
		return self.data.loc[ind, col]

class routeData():
        data = pd.DataFrame( [
            ['4_N_NULL_STOP', '4_N_NULL_STOP', '250N', 1.0],
            ['250N', '250N', '239N', 1.0],
            ['239N', '239N', '235N', 1.0],
            ['235N', '235N', '234N', 1.0],
            ['234N', '234N', '423N', 1.0],
            ['423N', '423N', '420N', 1.0],
            ['420N', '420N', '419N', 1.0],
            ['419N', '419N', '418N', 1.0],
            ['418N', '418N', '640N', 1.0],
            ['640N', '640N', '635N', 1.0],
            ['635N', '635N', '631N', 1.0],
            ['631N', '631N', '629N', 1.0],
            ['629N', '629N', '626N', 1.0],
            ['626N', '626N', '621N', 1.0],
            ['621N', '621N', '416N', 1.0],
            ['416N', '416N', '415N', 1.0],
            ['415N', '415N', '414N', 1.0],
            ['414N', '414N', '413N', 1.0],
            ['413N', '413N', '412N', 1.0],
            ['412N', '412N', '411N', 1.0],
            ['411N', '411N', '410N', 1.0],
            ['410N', '410N', '409N', 1.0],
            ['409N', '409N', '408N', 1.0],
            ['408N', '408N', '407N', 1.0],
            ['407N', '407N', '406N', 1.0],
            ['406N', '406N', '405N', 1.0],
            ['405N', '405N', '402N', 1.0],
            ['402N', '402N', '401N', 1.0],
            ['401N', '401N', '4_N_FINAL_STOP', 1.0]
        ] )
        data.columns = ['id','origin','destination','travel_time']
	data = data.set_index('id')

	def __getitem__(self, (ind, col)):
		return self.data.loc[ind, col]

def _null_tag_for(trip_id):
    ll = trip_id.split("_")[1][0]
    dd = trip_id.split(".")[-1][0]
    null_string = ll + "_" + dd + "_NULL_STOP"
    return null_string

class systemManager():
	def __init__(self):
                self.stopSeries = pd.Series( {
                    "250N" :stopObj("250N"),
                    "239N" :stopObj("239N"),
                    "235N" :stopObj("235N"),
                    "234N" :stopObj("234N"),
                    "423N" :stopObj("423N"),
                    "420N" :stopObj("420N"),
                    "419N" :stopObj("419N"),
                    "418N" :stopObj("418N"),
                    "640N" :stopObj("640N"),
                    "635N" :stopObj("635N"),
                    "631N" :stopObj("631N"),
                    "629N" :stopObj("629N"),
                    "626N" :stopObj("626N"),
                    "621N" :stopObj("621N"),
                    "416N" :stopObj("416N"),
                    "415N" :stopObj("415N"),
                    "414N" :stopObj("414N"),
                    "413N" :stopObj("413N"),
                    "412N" :stopObj("412N"),
                    "411N" :stopObj("411N"),
                    "410N" :stopObj("410N"),
                    "409N" :stopObj("409N"),
                    "408N" :stopObj("408N"),
                    "407N" :stopObj("407N"),
                    "406N" :stopObj("406N"),
                    "405N" :stopObj("405N"),
                    "402N" :stopObj("402N"),
                    "401N" :stopObj("401N")
                } )

                self.routeSeries = pd.Series( {
                    "4_N_NULL_STOP" :routeObj("4_N_NULL_STOP"),
                    "250N" :routeObj("250N"),
                    "239N" :routeObj("239N"),
                    "235N" :routeObj("235N"),
                    "234N" :routeObj("234N"),
                    "423N" :routeObj("423N"),
                    "420N" :routeObj("420N"),
                    "419N" :routeObj("419N"),
                    "418N" :routeObj("418N"),
                    "640N" :routeObj("640N"),
                    "635N" :routeObj("635N"),
                    "631N" :routeObj("631N"),
                    "629N" :routeObj("629N"),
                    "626N" :routeObj("626N"),
                    "621N" :routeObj("621N"),
                    "416N" :routeObj("416N"),
                    "415N" :routeObj("415N"),
                    "414N" :routeObj("414N"),
                    "413N" :routeObj("413N"),
                    "412N" :routeObj("412N"),
                    "411N" :routeObj("411N"),
                    "410N" :routeObj("410N"),
                    "409N" :routeObj("409N"),
                    "408N" :routeObj("408N"),
                    "407N" :routeObj("407N"),
                    "406N" :routeObj("406N"),
                    "405N" :routeObj("405N"),
                    "402N" :routeObj("402N"),
                    "401N" :routeObj("401N")
                } )

                self.activeTrains = {}
                self.selectLines = ['4']
                self.selectDirections = ['N']

		print "Stop Data Loaded",self.stopSeries.index

        def selectData(self, newDF):
                newDF['line'] = newDF['trip_id'].map(lambda x: x.split("_")[1][0]) 
                #newDF['direction'] = newDF['trip_id'].map(lambda x: x.split(".")[-1][0]) 
                newDF = newDF[newDF['line'].isin(self.selectLines)]
                newDF = newDF[newDF['stop'].isin(self.stopSeries.index)]
                return newDF

        def streamUpdate(self, newDF):
                newDF = self.selectData(newDF)
                print "culled stream data"
                print newDF[['timestamp', 'trip_id','stop','arrive']]
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
		scatterData = pd.DataFrame(index=[index_list], columns=['x', 'y', 'color', 'size', 'name'])
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
        		self.routeSeries[_null_tag_for(trip_id)].addTrain(trip_id, timestamp, t_arrive)
			print "new train",self.activeTrains[trip_id]['id']

		require_route_update, old_route_id, new_route_id = \
                        self.activeTrains[trip_id].update_trip(self, timestamp, next_stop, t_arrive)

                if require_route_update:
                        self.routeSeries[old_route_id].clearTrain(trip_id, timestamp)
                        self.routeSeries[new_route_id].addTrain(trip_id, timestamp, t_arrive)

class plotDataObj():
	def __init__(self):
		DF = pd.DataFrame(columns = ['x', 'y', 'color', 'size', 'name'])

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
                self.attrib['routeID'] = _null_tag_for(train_id)
		self.attrib['next_stop'] = _null_tag_for(train_id)
		self.attrib['sched_arrival'] = 0.
		self.attrib['trip_origin'] = self._calc_trip_origin()
		self.attrib['last_stop_time'] = 0.
                self.attrib['isLate'] = False
                self.attrib['name'] = self._make_train_name()

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
		quickPlotData = pd.DataFrame(index=[self['id']], columns=['x','y','color','size','name'], data=[[coords[0], coords[1], self._calc_color(), float(15), self['name']]])
		self.setPlotData(quickPlotData)

        def _calc_trip_origin(self):
                return int(self.attrib['id'].split("_")[0])

        def _calc_color(self):
                lateFlag = self.attrib['isLate']
                return {True:'red', False:'green'}[lateFlag]

        def _make_train_name(self):
                ll = self['id'].split("_")[1][0]
                dd = self['id'].split(".")[-1][0]
                name_string = ll + " Train " + {'N':"Uptown", 'S':"Downtown"}[dd]
                return name_string

class stopObj(vizComponent):
	def __init__(self, stop_id):
		vizComponent.__init__(self)
		self.stationLoc = stationLoc()
		self.attrib['id'] = stop_id
		self.attrib['lat'] = float(self.stationLoc[stop_id, 'lat'])
		self.attrib['lon'] = float(self.stationLoc[stop_id, 'lon'])
		#self.attrib['grid'] = np.array(self.stationLoc[stop_id, 'rel_grid'])
		self.attrib['name'] = np.array(self.stationLoc[stop_id, 'name'])
		#self.setPlotData(pd.DataFrame(index=[self['id']], columns=['x','y','color','size'], data=[[float(self.attrib['grid'][0]), float(self.attrib['grid'][1]), 'blue', float(10)]]))
		self.setPlotData(pd.DataFrame(index=[self['id']], columns=['x','y','color','size','name'], data=[[float(self.attrib['lon']), float(self.attrib['lat']), 'blue', float(25), self['name']]]))

class routeObj(vizComponent):
	def __init__(self, route_id):
		vizComponent.__init__(self)
                self.routeData = routeData()
                self.stationLoc = stationLoc()
		self.attrib['id'] = route_id
                self.attrib['origin_stop'] = self.routeData[route_id, 'origin']
                self.attrib['dest_stop'] = self.routeData[route_id, 'destination']
                self.attrib['travel_time'] = self.routeData[route_id, 'travel_time']
                
                #self.origin_coord = np.array(self.stationLoc[self.attrib['origin_stop'], 'rel_grid'])
                #self.dest_coord = np.array(self.stationLoc[self.attrib['dest_stop'], 'rel_grid'])
                origin_lon, origin_lat = self.stationLoc[self.attrib['origin_stop'],'lon'], self.stationLoc[self.attrib['origin_stop'], 'lat']
                dest_lon, dest_lat = self.stationLoc[self.attrib['dest_stop'],'lon'], self.stationLoc[self.attrib['dest_stop'], 'lat']
                print "lon/lat coords"
                print origin_lon, origin_lat
                print dest_lon, dest_lat
                self.origin_coord = np.array((origin_lon, origin_lat))
                self.dest_coord = np.array((dest_lon, dest_lat))
                print self.origin_coord
                print self.dest_coord
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
