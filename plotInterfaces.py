from bokeh.plotting import *
from bokeh.objects import Range1d, HoverTool
from collections import OrderedDict

SERVER_URL = """http://104.131.255.76:5006/"""

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

        def init_hover(self):
                self.TOOLS = ['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover']

	def init_plot(self, data, timestring):
		#figure(x_range = Range1d(start=-0.5, end=3.5))
                self.init_hover()

                source = ColumnDataSource(
                        data=dict(
                                x=data['x'],
                                y=data['y'],
                                label=data['name']
                        )
                )

		hold()

		scatter(data['x'], y=data['y'], alpha=0.3, color=data['color'], size=data['size'], source=source, tools=self.TOOLS)
                #text(x, y, text=inds, alpha=0.5, text_font_size="5pt", text_baseline="middle", text_align="center", angle=0)

                hover = [tools for tools in curplot().tools if isinstance(tools, HoverTool)][0]
                hover.tooltips = OrderedDict([
                        ("index", "$index"),
                        ("name", "@label"),
                        ("lon", "$x"),
                        ("lat", "$y")
                ])
	

                self.curplot = curplot
		self.curplot().title = "Subway Visualization " + str(timestring)
		#self.curplot().x_range = Range1d(start=0, end=4)
		#self.curplot().y_range = Range1d(start=0, end=4)
		#self.title = curplot().title
		#xaxis().major_label_orientation = np.pi/4
		xaxis().axis_label = "Subway Stop"
		yaxis().axis_label = ""
		show()
		self.renderer = [r for r in curplot().renderers if isinstance(r, Glyph)][0]
		self.ds = self.renderer.data_source
		self._first_plot = False

	def animate_plot(self, data, timestring):
		self.ds.data['x'] = data['x']
		self.ds.data['y'] = data['y']
		self.ds.data['size'] = data['size']
		self.ds.data['line_color'] = data['color']
		self.ds.data['fill_color'] = data['color']
		self.ds.data['name'] = data['name']

        	cursession().store_objects(self.ds)
		print "try to update plot title",str(timestring)
		curplot().title = "Subway Visualization " + str(timestring)
	
	def plot(self, data, timestring):
		if self._first_plot:
			self.init_plot(data, timestring)
		else:
			self.animate_plot(data, timestring)

