import pandas as pd
from bokeh.plotting import *
from bokeh.objects import Range1d, HoverTool
from collections import OrderedDict
import time

SERVER_URL = """http://localhost:5006"""

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

        def _init_hover(self, data, fields):
                if self._hover_enabled:
                        self.TOOLS = ['pan', 'wheel_zoom', 'box_zoom', 'resize', 'reset', 'hover']
                else:
                        self.TOOLS = ['pan', 'wheel_zoom', 'box_zoom', 'resize', 'reset']

		#choosefields = {'train':['name','approaching','duration','late'], 'station':['name','x','y']}
		columnDict = {}
		hoverlist = []
		for f in fields:
			columnDict[f] = data[f].values
			hoverlist.append(("\"" + f + "\"" , "\"@" + f + "\""))

                self.source = ColumnDataSource(data=columnDict)
		self.hoverDict = OrderedDict(hoverlist)

	def init_area(self, (xmin, xmax, ymin, ymax)):
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		print "ESTABLISHED PLOT BOUNDARIES"
		print self.xmin, self.xmax
		print self.ymin, self.ymax

	def _init_plot(self, allData, lineData, hoverFields, timestring):
                self._init_hover(allData, hoverFields)
		#staticplot = self.static_plot(staticData)

                #self.xmin = data['x'].min()
                #self.ymax = data['y'].max()

                figure(x_range=Range1d(start=self.xmin, end=self.xmax),
			y_range=Range1d(start=self.ymin, end=self.ymax),
			x_axis_type=None,
			y_axis_type=None,
                       	min_border=0,
                       	outline_line_color=None)

		hold()

		scatter(allData['x'], y=allData['y'], alpha=allData['alpha'], color=allData['color'], size=allData['size'], source=self.source, tools=self.TOOLS)

       	        multi_line(lineData['x'], lineData['y'], alpha=lineData['alpha'], color=lineData['color'])

                #text([self.xmin], [self.ymax], text=time.ctime(timestring), text_baseline="middle", text_align="left", angle=0)

                # get hold of this to display hover data
                hover = [tools for tools in curplot().tools if isinstance(tools, HoverTool)][0]
		hover.tooltips = self.hoverDict

                self.curplot = curplot

		curplot().title = "Subway Visualization"
		xaxis().grid_line_color = None
		yaxis().grid_line_color = None
		show()

                # get hold of this to refresh data for animation
		self.renderer = [r for r in curplot().renderers if isinstance(r, Glyph)][0]
		self.ds = self.renderer.data_source


		self._first_plot = False

	def _animate_plot(self, data, lineData, fields, timestring):
		for f in fields:
			self.ds.data[f] = data[f]

                #text([self.xmin], [self.ymax], text=time.ctime(timestring), text_baseline="middle", text_align="left", angle=0)

        	cursession().store_objects(self.ds)
	
	def plot(self, staticData, dynamicData, lineData, dynamicFields, hoverFields, timestring):
		df = pd.concat([staticData, dynamicData], axis=0)
		if self._first_plot:
			self._init_plot(df, lineData, hoverFields, timestring)
		else:
			self._animate_plot(df, lineData, dynamicFields, timestring)

