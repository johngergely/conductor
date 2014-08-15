import pandas as pd

class stationLoc():
        data = pd.read_csv("stops_formatted.txt", index_col=0)

	def get_area(self):
		self.lat_max = self.data['lat'].max()
		self.lat_min = self.data['lat'].min()
		self.lon_max = self.data['lon'].max()
		self.lon_min = self.data['lon'].min()
		return self.lon_min, self.lon_max, self.lat_min, self.lat_max

	def __getitem__(self, (ind, col)):
		return self.data.loc[ind, col]

## routes change dynamaically, so really these should be constructed from the real-time data
class routeData():
        data = pd.DataFrame( [
            ['4', 'N', '4_N_NULL_STOP', '4_N_NULL_STOP', '250N', 1.0],
            ['4', 'N', '250N', '250N', '239N', 1.0],
            ['4', 'N', '239N', '239N', '235N', 1.0],
            ['4', 'N', '235N', '235N', '234N', 1.0],
            ['4', 'N', '234N', '234N', '423N', 1.0],
            ['4', 'N', '423N', '423N', '420N', 1.0],
            ['4', 'N', '420N', '420N', '419N', 1.0],
            ['4', 'N', '419N', '419N', '418N', 1.0],
            ['4', 'N', '418N', '418N', '640N', 1.0],
            ['4', 'N', '640N', '640N', '635N', 1.0],
            ['4', 'N', '635N', '635N', '631N', 1.0],
            ['4', 'N', '631N', '631N', '629N', 1.0],
            ['4', 'N', '629N', '629N', '626N', 1.0],
            ['4', 'N', '626N', '626N', '621N', 1.0],
            ['4', 'N', '621N', '621N', '416N', 1.0],
            ['4', 'N', '416N', '416N', '415N', 1.0],
            ['4', 'N', '415N', '415N', '414N', 1.0],
            ['4', 'N', '414N', '414N', '413N', 1.0],
            ['4', 'N', '413N', '413N', '412N', 1.0],
            ['4', 'N', '412N', '412N', '411N', 1.0],
            ['4', 'N', '411N', '411N', '410N', 1.0],
            ['4', 'N', '410N', '410N', '409N', 1.0],
            ['4', 'N', '409N', '409N', '408N', 1.0],
            ['4', 'N', '408N', '408N', '407N', 1.0],
            ['4', 'N', '407N', '407N', '406N', 1.0],
            ['4', 'N', '406N', '406N', '405N', 1.0],
            ['4', 'N', '405N', '405N', '402N', 1.0],
            ['4', 'N', '402N', '402N', '401N', 1.0],
            ['4', 'N', '401N', '401N', '4_N_FINAL_STOP', 1.0],
            ['4', 'N', '4_S_NULL_STOP', '4_S_NULL_STOP', '401S', 1.0],
            ['4', 'S', '401S', '401S', '402S', 1.0],
            ['4', 'S', '402S', '402S', '405S', 1.0],
            ['4', 'S', '405S', '405S', '406S', 1.0],
            ['4', 'S', '406S', '406S', '407S', 1.0],
            ['4', 'S', '407S', '407S', '408S', 1.0],
            ['4', 'S', '408S', '408S', '409S', 1.0],
            ['4', 'S', '409S', '409S', '410S', 1.0],
            ['4', 'S', '410S', '410S', '411S', 1.0],
            ['4', 'S', '411S', '411S', '412S', 1.0],
            ['4', 'S', '412S', '412S', '413S', 1.0],
            ['4', 'S', '413S', '413S', '414S', 1.0],
            ['4', 'S', '414S', '414S', '415S', 1.0],
            ['4', 'S', '415S', '415S', '416S', 1.0],
            ['4', 'S', '416S', '416S', '621S', 1.0],
            ['4', 'S', '621S', '621S', '626S', 1.0],
            ['4', 'S', '626S', '626S', '629S', 1.0],
            ['4', 'S', '629S', '629S', '631S', 1.0],
            ['4', 'S', '631S', '631S', '635S', 1.0],
            ['4', 'S', '635S', '635S', '640S', 1.0],
            ['4', 'S', '640S', '640S', '418S', 1.0],
            ['4', 'S', '418S', '418S', '419S', 1.0],
            ['4', 'S', '419S', '419S', '420S', 1.0],
            ['4', 'S', '420S', '420S', '423S', 1.0],
            ['4', 'S', '423S', '423S', '234S', 1.0],
            ['4', 'S', '234S', '234S', '235S', 1.0],
            ['4', 'S', '235S', '235S', '239S', 1.0],
            ['4', 'S', '239S', '239S', '250S', 1.0],
            ['4', 'S', '250S', '250S', '4_S_FINAL_STOP', 1.0],
            ['5', 'N', '5_N_NULL_STOP', '5_N_NULL_STOP', '247N', 1.0],
            ['5', 'N', '247N', '247N', '246N', 1.0],
            ['5', 'N', '246N', '246N', '245N', 1.0],
            ['5', 'N', '245N', '245N', '244N', 1.0],
            ['5', 'N', '244N', '244N', '243N', 1.0],
            ['5', 'N', '243N', '243N', '242N', 1.0],
            ['5', 'N', '242N', '242N', '241N', 1.0],
            ['5', 'N', '241N', '241N', '239N', 1.0],
            ['5', 'N', '239N', '239N', '235N', 1.0],
            ['5', 'N', '235N', '235N', '234N', 1.0],
            ['5', 'N', '234N', '234N', '423N', 1.0],
            ['5', 'N', '423N', '423N', '420N', 1.0],
            ['5', 'N', '420N', '420N', '419N', 1.0],
            ['5', 'N', '419N', '419N', '418N', 1.0],
            ['5', 'N', '418N', '418N', '640N', 1.0],
            ['5', 'N', '640N', '640N', '635N', 1.0],
            ['5', 'N', '635N', '635N', '631N', 1.0],
            ['5', 'N', '631N', '631N', '629N', 1.0],
            ['5', 'N', '629N', '629N', '626N', 1.0],
            ['5', 'N', '626N', '626N', '621N', 1.0],
            ['5', 'N', '621N', '621N', '222N', 1.0],
            ['5', 'N', '222N', '222N', '221N', 1.0],
            ['5', 'N', '221N', '221N', '220N', 1.0],
            ['5', 'N', '220N', '220N', '219N', 1.0],
            ['5', 'N', '219N', '219N', '218N', 1.0],
            ['5', 'N', '218N', '218N', '217N', 1.0],
            ['5', 'N', '217N', '217N', '216N', 1.0],
            ['5', 'N', '216N', '216N', '215N', 1.0],
            ['5', 'N', '215N', '215N', '214N', 1.0],
            ['5', 'N', '214N', '214N', '213N', 1.0],
            ['5', 'N', '213N', '213N', '505N', 1.0],
            ['5', 'N', '505N', '505N', '504N', 1.0],
            ['5', 'N', '504N', '504N', '503N', 1.0],
            ['5', 'N', '503N', '503N', '502N', 1.0],
            ['5', 'N', '502N', '502N', '501N', 1.0],
            ['5', 'N', '501N', '501N', '5_N_FINAL_STOP', 1.0],
            ['5', 'S', '5_S_NULL_STOP', '5_S_NULL_STOP', '501S', 1.0],
            ['5', 'S', '501S', '501S', '502S', 1.0],
            ['5', 'S', '502S', '502S', '503S', 1.0],
            ['5', 'S', '503S', '503S', '504S', 1.0],
            ['5', 'S', '504S', '504S', '505S', 1.0],
            ['5', 'S', '505S', '505S', '213S', 1.0],
            ['5', 'S', '213S', '213S', '214S', 1.0],
            ['5', 'S', '214S', '214S', '215S', 1.0],
            ['5', 'S', '215S', '215S', '216S', 1.0],
            ['5', 'S', '216S', '216S', '217S', 1.0],
            ['5', 'S', '217S', '217S', '218S', 1.0],
            ['5', 'S', '218S', '218S', '219S', 1.0],
            ['5', 'S', '219S', '219S', '220S', 1.0],
            ['5', 'S', '220S', '220S', '221S', 1.0],
            ['5', 'S', '221S', '221S', '222S', 1.0],
            ['5', 'S', '222S', '222S', '621S', 1.0],
            ['5', 'S', '621S', '621S', '626S', 1.0],
            ['5', 'S', '626S', '626S', '629S', 1.0],
            ['5', 'S', '629S', '629S', '631S', 1.0],
            ['5', 'S', '631S', '631S', '635S', 1.0],
            ['5', 'S', '635S', '635S', '640S', 1.0],
            ['5', 'S', '640S', '640S', '418S', 1.0],
            ['5', 'S', '418S', '418S', '419S', 1.0],
            ['5', 'S', '419S', '419S', '420S', 1.0],
            ['5', 'S', '420S', '420S', '423S', 1.0],
            ['5', 'S', '423S', '423S', '234S', 1.0],
            ['5', 'S', '234S', '234S', '235S', 1.0],
            ['5', 'S', '235S', '235S', '239S', 1.0],
            ['5', 'S', '239S', '239S', '241S', 1.0],
            ['5', 'S', '241S', '241S', '242S', 1.0],
            ['5', 'S', '242S', '242S', '243S', 1.0],
            ['5', 'S', '243S', '243S', '244S', 1.0],
            ['5', 'S', '244S', '244S', '245S', 1.0],
            ['5', 'S', '245S', '245S', '246S', 1.0],
            ['5', 'S', '246S', '246S', '247S', 1.0],
            ['5', 'S', '247S', '247S', '5_S_FINAL_STOP', 1.0],
            ['6', 'N', '6_N_NULL_STOP', '6_N_NULL_STOP', '640N', 1.0],
            ['6', 'N', '640N', '640N', '639N', 1.0],
            ['6', 'N', '639N', '639N', '638N', 1.0],
            ['6', 'N', '638N', '638N', '637N', 1.0],
            ['6', 'N', '637N', '637N', '636N', 1.0],
            ['6', 'N', '636N', '636N', '635N', 1.0],
            ['6', 'N', '635N', '635N', '634N', 1.0],
            ['6', 'N', '634N', '634N', '633N', 1.0],
            ['6', 'N', '633N', '633N', '632N', 1.0],
            ['6', 'N', '632N', '632N', '631N', 1.0],
            ['6', 'N', '631N', '631N', '630N', 1.0],
            ['6', 'N', '630N', '630N', '629N', 1.0],
            ['6', 'N', '629N', '629N', '628N', 1.0],
            ['6', 'N', '628N', '628N', '627N', 1.0],
            ['6', 'N', '627N', '627N', '626N', 1.0],
            ['6', 'N', '626N', '626N', '625N', 1.0],
            ['6', 'N', '625N', '625N', '624N', 1.0],
            ['6', 'N', '624N', '624N', '623N', 1.0],
            ['6', 'N', '623N', '623N', '622N', 1.0],
            ['6', 'N', '622N', '622N', '621N', 1.0],
            ['6', 'N', '621N', '621N', '619N', 1.0],
            ['6', 'N', '619N', '619N', '618N', 1.0],
            ['6', 'N', '618N', '618N', '617N', 1.0],
            ['6', 'N', '617N', '617N', '616N', 1.0],
            ['6', 'N', '616N', '616N', '615N', 1.0],
            ['6', 'N', '615N', '615N', '614N', 1.0],
            ['6', 'N', '614N', '614N', '613N', 1.0],
            ['6', 'N', '613N', '613N', '612N', 1.0],
            ['6', 'N', '612N', '612N', '611N', 1.0],
            ['6', 'N', '611N', '611N', '610N', 1.0],
            ['6', 'N', '610N', '610N', '609N', 1.0],
            ['6', 'N', '609N', '609N', '608N', 1.0],
            ['6', 'N', '608N', '608N', '607N', 1.0],
            ['6', 'N', '607N', '607N', '606N', 1.0],
            ['6', 'N', '606N', '606N', '604N', 1.0],
            ['6', 'N', '604N', '604N', '603N', 1.0],
            ['6', 'N', '603N', '603N', '602N', 1.0],
            ['6', 'N', '602N', '602N', '601N', 1.0],
            ['6', 'N', '601N', '601N', '6_N_FINAL_STOP', 1.0],
            ['6', 'S', '6_S_NULL_STOP', '6_S_NULL_STOP', '601S', 1.0],
            ['6', 'S', '601S', '601S', '602S', 1.0],
            ['6', 'S', '602S', '602S', '603S', 1.0],
            ['6', 'S', '603S', '603S', '604S', 1.0],
            ['6', 'S', '604S', '604S', '606S', 1.0],
            ['6', 'S', '606S', '606S', '607S', 1.0],
            ['6', 'S', '607S', '607S', '608S', 1.0],
            ['6', 'S', '608S', '608S', '609S', 1.0],
            ['6', 'S', '609S', '609S', '610S', 1.0],
            ['6', 'S', '610S', '610S', '611S', 1.0],
            ['6', 'S', '611S', '611S', '612S', 1.0],
            ['6', 'S', '612S', '612S', '613S', 1.0],
            ['6', 'S', '613S', '613S', '614S', 1.0],
            ['6', 'S', '614S', '614S', '615S', 1.0],
            ['6', 'S', '615S', '615S', '616S', 1.0],
            ['6', 'S', '616S', '616S', '617S', 1.0],
            ['6', 'S', '617S', '617S', '618S', 1.0],
            ['6', 'S', '618S', '618S', '619S', 1.0],
            ['6', 'S', '619S', '619S', '621S', 1.0],
            ['6', 'S', '621S', '621S', '622S', 1.0],
            ['6', 'S', '622S', '622S', '623S', 1.0],
            ['6', 'S', '623S', '623S', '624S', 1.0],
            ['6', 'S', '624S', '624S', '625S', 1.0],
            ['6', 'S', '625S', '625S', '626S', 1.0],
            ['6', 'S', '626S', '626S', '627S', 1.0],
            ['6', 'S', '627S', '627S', '628S', 1.0],
            ['6', 'S', '628S', '628S', '629S', 1.0],
            ['6', 'S', '629S', '629S', '630S', 1.0],
            ['6', 'S', '630S', '630S', '631S', 1.0],
            ['6', 'S', '631S', '631S', '632S', 1.0],
            ['6', 'S', '632S', '632S', '633S', 1.0],
            ['6', 'S', '633S', '633S', '634S', 1.0],
            ['6', 'S', '634S', '634S', '635S', 1.0],
            ['6', 'S', '635S', '635S', '636S', 1.0],
            ['6', 'S', '636S', '636S', '637S', 1.0],
            ['6', 'S', '637S', '637S', '638S', 1.0],
            ['6', 'S', '638S', '638S', '639S', 1.0],
            ['6', 'S', '639S', '639S', '640S', 1.0],
            ['6', 'S', '640S', '640S', '6_S_FINAL_STOP', 1.0]
        ] )
        data.columns = ['line','direction','id','origin','destination','travel_time']
        data['index_col'] = data['origin'] + "_" + data['destination']
	data = data.set_index('index_col')

        def get(self, line, direction):
                sliceDF = self.data[self.data['line'] == line]
                sliceDF = sliceDF[sliceDF['direction'] == direction]
                return sliceDF

	def __getitem__(self, (ind, col)):
		return self.data.loc[ind, col]