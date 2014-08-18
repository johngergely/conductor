import numpy as np
import pandas as pd
import math
import time

from stations import station_names
from collect import get_TOD_reference, nice_time
from systemData import stationLoc, routeData
from aggregator import aggregator, NULL_STATS
from visualize import file_timestamp

ONE_BY_SIXTY = 1./60
line_mult = 3

# color palette
BRICK = "#800000"
BLUE = "#0066CC"
GOLDENROD = "#FFCC00"
LT_YELLOW = "#FFE066"
LATE_MAGENTA = "#FF0066"
GREEN = "#009933"

def unit_perp(U, root_sign=1.):
    ux = U[0]
    uy = U[1]
    xi = ux*ux/(uy*uy)
    vx = math.sqrt(xi/(1.+xi))
    if root_sign*vx < ux:
	    vx = -1.*vx
    vy = -ux*vx/uy
    return np.array((vx, vy))

def _null_tag_for(trip_id):
    ll = trip_id.split("_")[1][0]
    dd = trip_id.split(".")[-1][0]
    null_string = ll + "_" + dd + "_NULL_STOP"
    return null_string

def _make_string(data_dict):
	S = ""
	for k in data_dict.keys():
		S = S + k + " : " + str(data_dict[k]) + '\n'
	return S

class systemManager():
	def __init__(self, setLines, setDirections):
                self.selectLines = setLines
                self.selectDirections = setDirections
                self.routeData = routeData()
                self.stationLoc = stationLoc()

                # for stops we need only use 'N' objects
                stopList = []
                for ll in ['4','5','6']:
                    for ii in self.routeData.get(ll, "N")['id']:
                        stopList.append(ii)
                stopList = set(stopList)
                self.stopSeries = pd.Series(index=stopList)
                for si in self.stopSeries.index:
                    self.stopSeries[si] = stopObj(si, self.stationLoc[si,:])

                self._activeRoutes = {}
		#self._routestats = {}
                self._allRoutes = {}
                for ll in ['4','5','6']:
                    for dd in ['N','S']:
                        routeSlice = self.routeData.get(ll, dd)
			fname_string = "data/subway_data_" + file_timestamp + "_" + ll + "_" + dd + "_stoptimes.csv"
			agg = aggregator(fname_string)
			#agg.select_range(--criteria--)
			#agg.reset()
                        for ri in routeSlice.index:
                                rData = routeSlice.loc[ri,:]
                                #print "instantiating route",ri,rData
				routeStats = agg.process_tuple(rData['origin'], rData['destination'])
                                self._allRoutes[ri] = routeObj(ri, rData['origin'], rData['destination'], self.stationLoc, stats=routeStats)
                print "INITIALIZED ALL ROUTES",len(self._allRoutes)

                self.activeTrains = {}
		#print "Stop Data Loaded",self.stopSeries.index

	def plot_boundaries(self):
		xmin, xmax, ymin, ymax = self.stationLoc.get_area()
		return xmin, xmax, ymin, ymax

        def selectData(self, newDF):
                newDF['line'] = newDF['trip_id'].map(lambda x: x.split("_")[1][0]) 
                newDF['direction'] = newDF['trip_id'].map(lambda x: x.split(".")[-1][0]) 
                newDF = newDF[newDF['line'].isin(self.selectLines)]
                #newDF = newDF[newDF['stop'].isin(self.stopSeries.index)]
                newDF = newDF[newDF['direction'].isin(self.selectDirections)]
                return newDF

        def streamUpdate(self, newDF):
                newDF = self.selectData(newDF)
                #print "culled stream data"
                #print newDF[['timestamp', 'trip_id','stop','arrive']]
                for i in newDF.index:
                        self._updateTrain(newDF.loc[i, 'trip_id'],
                                         newDF.loc[i, 'timestamp'],
                                         newDF.loc[i,'stop'],
                                         newDF.loc[i,'arrive'],
					 newDF.loc[i,'depart'])
               
        def evolve(self, t_now, t_ref):
                for train in self.activeTrains.values():
                        routeOrigin, routeDest = train.attrib['routeID']
			train_id = train.attrib['id']
                        coords, isLate = self._getRoute((routeOrigin, routeDest)).trainPosition(train_id, t_now)
		        train.update_position(t_now, coords, isLate)

	def drawSystem(self, timestring):
		fields = ['name', 'x', 'y', 'color', 'size', 'alpha', 'info']
		index_list = list(self.stopSeries.index.values)
		stationData = pd.DataFrame(index=[index_list], columns=fields)
		for stop in self.stopSeries.index:
			stationData.loc[stop,:] = self.stopSeries[stop].data().loc[stop,:]

		index_list = []
		for k in self.activeTrains.keys():
			if self.activeTrains[k].attrib['active_trip']:
				index_list.append(k)
		scatterData = pd.DataFrame(index=[index_list], columns=fields)
		for train in self.activeTrains.keys():
			scatterData.loc[train,:] = self.activeTrains[train].data().loc[train,:]

		lineData = {'x':[], 'y':[], 'alpha':[], 'color':[], 'line_width':[]}
                for route in self._allRoutes.values():
                    for ii in lineData.keys():
			    for route_index in route.data().index:
                            	lineData[ii].append(route.data().loc[route_index,ii])
		return stationData, scatterData, lineData, fields, ['name','info']

	def _updateTrain(self, trip_id, timestamp, next_stop, t_arrive, t_depart):
		t_arrive = max(t_arrive, t_depart)
		if not self.activeTrains.get(trip_id):
                        prev_stop = self._lookupPrev(trip_id, next_stop)
        		self.activeTrains[trip_id] = trainObj(trip_id, prev_stop, next_stop, timestamp)
        		self._getRoute((prev_stop, next_stop)).addTrain(trip_id, timestamp, t_arrive)
			#print "new train",self.activeTrains[trip_id]['id'],prev_stop, next_stop
                        #print "active on route",self._getRoute((prev_stop, next_stop)).activeTrains.keys(),prev_stop,next_stop

		require_route_update, old_route_tuple, new_route_tuple = \
                        self.activeTrains[trip_id].update_trip(timestamp, next_stop, t_arrive)

                if require_route_update:
                        self._getRoute(old_route_tuple).clearTrain(trip_id, timestamp)
                        self._getRoute(new_route_tuple).addTrain(trip_id, timestamp, t_arrive)

        def _getRoute(self, (origin, dest)):
            tag = origin + "_" + dest
            if not self._allRoutes.get(tag):
	        print "_getRoute constructor",origin,dest
                self._allRoutes[tag] = routeObj(tag, origin, dest, self.stationLoc)
            #return self._activeRoutes[tag]
            return self._allRoutes[tag]

        def _lookupPrev(self, trip_id, next_stop):
            result = self.routeData.data[self.routeData.data['destination']==next_stop]
            if len(result) == 0:
                #null_tag = _null_tag_for(trip_id)
                print "LOOKUP_PREV FAILED",next_stop,"RETURNING THE DESTINATION"
                return next_stop
            return result['origin'].values[0]

class plotDataObj():
	def __init__(self):
		self.DF = pd.DataFrame(columns = ['x', 'y', 'color', 'size', 'name'])

	def setData(self, setDF):
		self.DF = setDF

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
	def __init__(self, train_id, prev_stop, next_stop, timestamp):
		vizComponent.__init__(self)
		self.attrib['id'] = train_id
		self.attrib['time_of_update'] = timestamp
                null_tag = _null_tag_for(train_id)
                self.attrib['next_stop'] = next_stop 
                self.attrib['prev_stop'] = prev_stop 
                self.attrib['routeID'] = (self.attrib['prev_stop'], self.attrib['next_stop'])
		self.attrib['sched_arrival'] = 0.
		self.attrib['trip_origin'] = self._calc_trip_origin()
		self.attrib['active_trip'] = self.attrib['trip_origin'] < self.attrib['time_of_update']
		self.attrib['last_stop_time'] = 0.
                self.attrib['isLate'] = False
                self.attrib['name'] = self._make_train_name()
		self.attrib['t_late'] = 0.
		self.attrib['duration'] = 0.
		self.update_count = 0

                #print "create train obj route/stop",self.attrib['routeID'],self.attrib['next_stop']

	def update_trip(self, time_of_update, next_stop, t_arrive):
		self.attrib['active_trip'] = self.attrib['trip_origin'] < self.attrib['time_of_update']
		self.attrib['time_of_update'] = time_of_update
		self.attrib['sched_arrival'] = t_arrive
		self.attrib['duration'] = (time_of_update - self.attrib['trip_origin'])/60.

                ## passed a stop; update params for next stop
                if next_stop != self.attrib['next_stop']:
                        self.attrib['last_stop_time'] = time_of_update
                        old_route_tuple = self.attrib['routeID']
                        self.attrib['prev_stop'] = self.attrib['next_stop']
		        self.attrib['next_stop'] = next_stop
                        self.attrib['routeID'] = (self.attrib['prev_stop'], self.attrib['next_stop'])
                        self.attrib['isLate'] = False

                        return True, old_route_tuple, self.attrib['routeID']
                else:
                        if time_of_update > t_arrive:
                                self.attrib['isLate'] = True
				self.attrib['t_late'] = (time_of_update - t_arrive)/60.
                        return False, self.attrib['routeID'], self.attrib['routeID']

        def update_position(self, timestamp, coords, isLate):
		self.update_count += 1
                self.attrib['isLate'] = isLate 
		if isLate:
			self.attrib['t_late'] = (timestamp - self.attrib['sched_arrival'])/60.
		## x, y, color, size, name, info(string contains approaching, duration, late)
		infoString = _make_string({
                        "time":nice_time(timestamp, military=False),
			"approaching":station_names[self['next_stop']],
			"duration":"%.1f" % float(self['duration']),
			"late":"%.1f" % float(self.attrib['t_late']),
			"update":self.update_count
			})
		quickPlotData = pd.DataFrame(index=[self['id']], columns=['x','y','color','size','alpha','name', 'info'],
			data=[[coords[0], coords[1], self._calc_color(), float(12), float(0.4),
				self['name'], infoString]]
			)
		self.setPlotData(quickPlotData)

        def _calc_trip_origin(self):
		t_ref = get_TOD_reference(float(self.attrib['time_of_update']))
		minutes100 = float(self.attrib['id'].split("_")[0])
		#print "CALC ORIGIN",0.6*minutes100 + t_ref,self.attrib["time_of_update"], self.attrib["time_of_update"]-(0.6*minutes100 + t_ref)
                return int(0.6*minutes100 + t_ref)

        def _calc_color(self):
                lateFlag = self.attrib['isLate']
                return {True:LATE_MAGENTA, False:GREEN}[lateFlag]

        def _make_train_name(self):
                ll = self['id'].split("_")[1][0]
                dd = self['id'].split(".")[-1][0]
                name_string = ll + " Train " + {'N':"Uptown", 'S':"Downtown"}[dd]
                return name_string

class stopObj(vizComponent):
	def __init__(self, stop_id, stopData):
		vizComponent.__init__(self)
		self.attrib['id'] = stop_id
		self.attrib['lat'] = float(stopData['lat'])
		self.attrib['lon'] = float(stopData['lon'])
		self.attrib['name'] = np.array(stopData['name'])
		#self.attrib['grid'] = np.array(stopData['rel_grid'])
		#self.setPlotData(pd.DataFrame(index=[self['id']], columns=['x','y','color','size'], data=[[float(self.attrib['grid'][0]), float(self.attrib['grid'][1]), 'blue', float(10)]]))
		infoString = _make_string({
			"lat" : float(self.attrib['lat']),
			"lon" : float(self.attrib['lon'])
			})
		self.setPlotData(pd.DataFrame(index=[self['id']], columns=['x','y','color','size','alpha','name','info'],
			data=[[float(self.attrib['lon']), float(self.attrib['lat']), BLUE, float(5), float(1.0), str(self['name']) + str(" Station"), infoString
				]]))

class routeObj(vizComponent):
	def __init__(self, route_id, origin_id, destination_id, stationLoc, travel_time=1.0, stats=NULL_STATS):
		vizComponent.__init__(self)
		self.attrib['id'] = route_id
                self.attrib['origin_stop'] = origin_id
                self.attrib['dest_stop'] = destination_id
                self.attrib['travel_time'] = travel_time 
		self.stats = stats
                
                #self.origin_coord = np.array(self.stationLoc[self.attrib['origin_stop'], 'rel_grid'])
                #self.dest_coord = np.array(self.stationLoc[self.attrib['dest_stop'], 'rel_grid'])
                origin_lon, origin_lat = stationLoc[self.attrib['origin_stop'],'lon'], stationLoc[self.attrib['origin_stop'], 'lat']
                dest_lon, dest_lat = stationLoc[self.attrib['dest_stop'],'lon'], stationLoc[self.attrib['dest_stop'], 'lat']
                self.origin_coord = np.array((origin_lon, origin_lat))
                self.dest_coord = np.array((dest_lon, dest_lat))
                #print "ROUTE",self['id'], self.origin_coord, self.dest_coord
                self.x_coords = np.array((origin_lon, dest_lon))
                self.y_coords = np.array((origin_lat, dest_lat))
                #print "ROUTE",self['id'], self.x_coords, self.y_coords
                self.activeTrains = {}
                infoString = "route"
		self.setPlotData(pd.DataFrame(index=[self['id']+"_base", self['id']+"_median", self['id']+"_75pct"],
			columns=['x','y','color','alpha','line_width','name','info'],
			data=[
				#[self.x_coords, self.y_coords, BLUE, float(1.0), line_mult*ONE_BY_SIXTY*self.stats['min'], str(self['id']), infoString],
				[self.x_coords, self.y_coords, BLUE, 1.0, 1.0, str(self['id']), infoString],
				[self.x_coords, self.y_coords, GOLDENROD, 0.2, line_mult*self.stats['50%']/self.stats['min'], str(self['id']), infoString],
				[self.x_coords, self.y_coords, LT_YELLOW, 0.2, line_mult*self.stats['75%']/self.stats['min'], str(self['id']), infoString]
			     ]
			))

        def trainPosition(self, trip_id, timestamp, dir_shift=True):
                t_start, t_arrive, isLate = self.activeTrains[trip_id]
                progress_fraction = max(0., float(timestamp - t_start)/(t_arrive - t_start))
                if progress_fraction > 0.95:
                        progress_fraction = 0.98
                        self.activeTrains[trip_id] = t_start, t_arrive, True
                        isLate = True
		#print trip_id,timestamp,t_start,t_arrive,self.origin_coord, self.dest_coord,progress_fraction

		U = self.dest_coord - self.origin_coord
		if np.dot(U,U) == 0.:
                    return self.origin_coord, isLate
		V = np.array((0., 0.))
		if dir_shift:
		    dd = trip_id.split(".")[-1][0]
		    sign = 1.
		    if dd=="S":
                        sign = -1.
		    V = unit_perp(U, sign)
		    #print "unit vector",V,np.dot(U,V),U
		updateCoord = self.origin_coord + progress_fraction*U + 0.0015*V
                return updateCoord, isLate

        def addTrain(self, trip_id, t_start, t_arrive):
		isLate = False
		if t_start > t_arrive:
			#print trip_id, "behind schedule", t_start, t_arrive
			#print "Using historical median",self.stats['50%']
			t_arrive = t_start + self.stats['50%']
			isLate = True
                self.activeTrains[trip_id] = (t_start, t_arrive, isLate)

        def clearTrain(self, trip_id, timestamp):
                self.activeTrains.pop(trip_id)
	
if __name__=="__main__":
        print "dataEngine::__main__"
