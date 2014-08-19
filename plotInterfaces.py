import pandas as pd
from bokeh.plotting import *
from bokeh.objects import Range1d, HoverTool
from bokeh.embed import autoload_server
from collections import OrderedDict

import time
import numpy as np

from dataEngine import unit_perp
from createLink import make_URL

SERVER_URL = """http://104.131.255.76:5006/"""

# color palette
BRICK = "#800000"
BLUE = "#0066CC"
GOLDENROD = "#FFCC00"
LT_YELLOW = "#FFE066"
LATE_MAGENTA = "#FF0066"
GREEN = "#009933"
GRAY = "#4C4E52"

def _make_polygon(U0, U1, a, m):
    V = unit_perp(U1-U0)
    x0 = U0[0]
    y0 = U0[1]
    x1 = U1[0]
    y1 = U1[1]
    b = 1. - a
    x_set = [x0, b*x0+a*x1+m*V[0], a*x0+b*x1+m*V[0], x1, a*x0+b*x1-m*V[0], b*x0+a*x1-m*V[0]]
    y_set = [y0, b*y0+a*y1+m*V[1], a*y0+b*y1+m*V[1], y1, a*y0+b*y1-m*V[1], b*y0+a*y1-m*V[1]]
    return x_set, y_set

def _plotLineData(lineData):
    xs = []
    ys = []
    alphas = []
    colors = []
    widths = []
    for i in range(len(lineData['x'])):
        xx = lineData['x'][i]
        yy = lineData['y'][i]
        tmin = float(lineData['t_min'][i])
        t50 = float(lineData['t_50pct'][i])
        t75 = float(lineData['t_75pct'][i])

        xs.append(xx)
        ys.append(yy)
        alphas.append(1.0)
        colors.append(BLUE)
        widths.append(1.0)
        ##xs.append(xx)
        ##ys.append(yy)
        ##alphas.append(0.25)
        ##colors.append(GOLDENROD)
        ##widths.append(3*t50/tmin)
        ##xs.append(xx)
        ##ys.append(yy)
        ##alphas.append(0.15)
        ##colors.append(LT_YELLOW)
        ##widths.append(3*t75/tmin)
    return multi_line(xs, ys, alpha=alphas, color=colors, line_width=widths, line_cap="round", line_join="round")

def _plotPatches(lineData):
    xs = []
    ys = []
    alphas = []
    colors = []
    widths = []
    x_scale = lineData['x'][1]-lineData['x'][0]
    print "spatial scale",x_scale
    for i in range(len(lineData['x'])):
        xx = lineData['x'][i]
        yy = lineData['y'][i]
        tmin = float(lineData['t_min'][i])
        t50 = float(lineData['t_50pct'][i])
        t75 = float(lineData['t_75pct'][i])

        U = np.array((xx[1]-xx[0], yy[1]-yy[0]))
	if np.dot(U,U) == 0.:
            continue
        else:
            r0 =  np.array((xx[0], yy[0]))
            r1 =  np.array((xx[1], yy[1]))
            w = 0.5 * t50/tmin * 0.02 * x_scale[0]
            x_set, y_set =  _make_polygon(r0, r1, 0.2, w)

            xs.append(x_set)
            ys.append(y_set)
            alphas.append([0.25])
            colors.append([GOLDENROD])

    return patches(xs, ys, fill_color=colors, fill_alpha=alphas, line_alpha=0.2, line_color=GOLDENROD)

class bokehPlotInterface():
	def __init__(self, plot_mode="server"):
		if plot_mode != "server":
			print "requested bokeh plot mode not set up yet",plot_mode
		else:
			output_server("plot.html", url=SERVER_URL)#, load_from_config=False)

		self.renderer = []
		self.ds = {}
		self._first_plot = True
                self._hover_enabled = True

	def init_area(self, (xmin, xmax, ymin, ymax)):
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		print "ESTABLISHED PLOT BOUNDARIES"
		print self.xmin, self.xmax
		print self.ymin, self.ymax

        def _init_hover(self, data, fields):
                if self._hover_enabled:
                        self.TOOLS = ['pan', 'box_zoom', 'resize', 'reset', 'hover']
                else:
                        self.TOOLS = ['pan', 'box_zoom', 'resize', 'reset']

		#choosefields = {'train':['name','approaching','duration','late'], 'station':['name','x','y']}
		columnDict = {}
		hoverlist = []
		for f in fields:
			columnDict[f] = data[f].values
			hoverlist.append(("\"" + f + "\"" , "\"@" + f + "\""))

                self.source = ColumnDataSource(data=columnDict)
		self.hoverDict = OrderedDict(hoverlist)

	def _init_plot(self, allData, lineData, hoverFields, timestring):
                self._init_hover(allData, hoverFields)
		#staticplot = self.static_plot(staticData)

                #self.xmin = data['x'].min()
                #self.ymax = data['y'].max()

                figure(x_range=Range1d(start=self.xmin, end=self.xmax),
			y_range=Range1d(start=self.ymin, end=self.ymax),
		        title = "CONDUCTOR - Real-time Subway Visualization and Analytics",
		        title_text_font_size = "14pt",
		        title_text_color = GRAY, 
			x_axis_type=None,
			y_axis_type=None,
                       	min_border=0,
                       	outline_line_color=None)

		hold()

		scatter(allData['x'], y=allData['y'], alpha=allData['alpha'], color=allData['color'], size=allData['size'], source=self.source, tools=self.TOOLS)
               
                _plotPatches(lineData)
                _plotLineData(lineData)

                #text([self.xmin], [self.ymax], text=time.ctime(timestring), text_baseline="middle", text_align="left", angle=0)

                # get hold of this to display hover data
                hover = [tools for tools in curplot().tools if isinstance(tools, HoverTool)][0]
		hover.tooltips = self.hoverDict

                self.curplot = curplot

		xaxis().grid_line_color = None
		yaxis().grid_line_color = None
		show()

                # get hold of this to refresh data for animation
		self.renderer = [r for r in curplot().renderers if isinstance(r, Glyph)][0]
		self.ds = self.renderer.data_source


		self._first_plot = False

                print "calling autoload_server"
                EMBED_DATA = autoload_server(curplot(), cursession())
                make_URL(SERVER_URL, EMBED_DATA)

	def _animate_plot(self, data, lineData, fields, timestring):
		for f in fields:
			self.ds.data[f] = data[f]
			#print data[f].values

                #text([self.xmin], [self.ymax], text=time.ctime(timestring), text_baseline="middle", text_align="left", angle=0)

        	cursession().store_objects(self.ds)
	
	def plot(self, staticData, dynamicData, lineData, dynamicFields, hoverFields, timestring):
		df = pd.concat([staticData, dynamicData], axis=0)
		if self._first_plot:
			self._init_plot(df, lineData, hoverFields, timestring)
		else:
			self._animate_plot(df, lineData, dynamicFields, timestring)

if __name__=="__main__":
	print "TESTING PLOT WITH DATA SET"
	#static = pd.DataFrame(index=[1,2,3], [[1,1],[2,2],[3,3]])
