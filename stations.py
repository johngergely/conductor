import pandas as pd
import numpy as np
from string import lower

S = pd.read_csv("stops.txt")

station_names = dict(zip(S['stop_id'],S['stop_name']))
station_codes = dict(zip(S['stop_name'],S['stop_id']))

code_order_4_N = ['250N','239N','235N', '234N','423N',
		'420N', '419N', '418N','640N',
		'635N','631N','629N','626N','621N',
		'416N', '415N', '414N','413N', '412N',
		'411N', '410N', '409N', '408N','407N',
		'406N','405N','402N', '401N']

code_order_4_S = [s[:-1]+'S' for s in reversed(code_order_4_N)]

code_order_5_N = ['247N','239N','235N', '234N','423N',
		'420N', '419N', '418N','640N',
		'635N','631N','629N','626N','621N',
		'222N','221N','220N','219N','218N','217N',
		'216N','215N','214N','213N',
		'505N','504N','503N','502N','501N']
code_order_5_S = [s[:-1]+'S' for s in reversed(code_order_5_N)]

code_order_6_N = ['640N','639N','638N', '637N','636N','635N',
		'634N', '633N', '632N', '631N', '630N',
		'629N', '628N', '627N', '626N', '625N', '624N',
		'623N', '622N', '621N', '619N', '618N',
		'617N', '616N', '615N', '614N', '613N', '612N',
		'611N', '610N', '609N', '608N', '607N', '606N',
		'604N', '603N', '602N', '601N']
code_order_6_S = [s[:-1]+'S' for s in reversed(code_order_6_N)]

code_order_dict = {"4_N" : code_order_4_N,
		"4_S" : code_order_4_S,
		"5_N" : code_order_5_N,
		"5_S" : code_order_5_S,
		"6_N" : code_order_6_N,
		"6_S" : code_order_6_S
		}

directions = ['N','S']
lines = ['4','5','6']

def get_station_map(direction, line):
	if direction not in directions:
		print direction,"not recognized; allowed directions are",directions
		return None
	if line not in lines:
		print line,"not recognized; allowed directions are",lines
		return None
	code_order = code_order_dict[line+"_"+direction]
	print "got codes list",code_order
	station_order = [station_names[s] for s in code_order]
	station_map = dict(zip(code_order, np.arange(len(code_order))))
	station_index = dict(zip(np.arange(len(code_order)), code_order))
	return station_order, code_order, station_map, station_index

DEFINE_STOPS_LEX_AVE_NORTHBOUND = {'Utica':'250N', 'Flatbush':'247N', 'Atlantic':'235N', 'Fulton':'418N', 'Brooklyn Bridge':'640N', 'Union Sq':'635N', 'Grand Central':'631N', '125 St':'621N', 'Woodlawn':'401N', 'Dyre':'501N', 'Pelham':'601N'}
DEFINE_STOPS_LEX_AVE_SOUTHBOUND = {'Utica':'250S', 'Flatbush':'247S', 'Atlantic':'235S', 'Fulton':'418S', 'Brooklyn Bridge':'640S', 'Union Sq':'635S', 'Grand Central':'631S', '125 St':'621S', 'Woodlawn':'401S', 'Dyre':'501S', 'Pelham':'601S'}

def _get_stops(direction, set_range="ALL", route="LEX"):
        if lower(route) != "lex":
                print "Right now only Lexington Avenue 4/5/6 routes are supported. You requested unsupported route parameter",route
                return None
        dir_flag = lower(direction[0]) 
        use_dict = DEFINE_STOPS_LEX_AVE_NORTHBOUND
        start_list = ['Utica', 'Flatbush', 'Brooklyn_Bridge']
        end_list = ['Woodlawn', 'Dyre', 'Pelham']
        others_list = [k for k in DEFINE_STOPS_LEX_AVE_NORTHBOUND.keys() if (k not in start_list and k not in end_list)]
        if dir_flag == "s":
                use_dict = DEFINE_STOPS_LEX_AVE_SOUTHBOUND
                start_list = ['Woodlawn', 'Dyre', 'Pelham']
                end_list = ['Utica', 'Flatbush', 'Brooklyn_Bridge']
        elif dir_flag != "n":
                print "Failed to return list of stops! Allowed direction parameters are [N]orthbound or [S]outhbound; You specified",direction
                return None
        if lower(set_range) == "all":
                return use_dict.values()
        elif lower(set_range) == "interim":
                return [use_dict.get(k) for k in others_list]
        elif lower(set_range) == "start":
                return [use_dict.get(k) for k in start_list]
        elif lower(set_range) == "end":
                return [use_dict.get(k) for k in end_list]
        else:
                allowed_range = ["all", "start", "end", "interim"]
                print "Failed to return list of stops! Allowed range parameters are",allowed_range,"; You specified",set_range
                return None

def get_stop_dict(direction, route="LEX"):
        """ Return the dictionary of station stops and stop codes."""
        if lower(route) != "lex":
                print "Right now only Lexington Avenue 4/5/6 routes are supported. You requested unsupported route parameter",route
                return None
        dir_flag = lower(direction[0]) 
        use_dict = DEFINE_STOPS_LEX_AVE_NORTHBOUND
        if dir_flag == "s":
                use_dict = DEFINE_STOPS_LEX_AVE_SOUTHBOUND
        elif dir_flag != "n":
                print "Failed to return list of stops! Allowed direction parameters are [N]orthbound or [S]outhbound; You specified",direction
                return None
        return use_dict

def get_line(trip_id):
    return trip_id.split("_")[1][0]

def stop_for(l,terminus):
    return {"N":{'4':'Woodlawn', '5':'Dyre', '6':'Pelham'}, "S":{'4':'Utica', '5':'Flatbush','6':'Brooklyn Bridge'}}[terminus][l]

