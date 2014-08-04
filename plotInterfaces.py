from bokeh.plotting import *
from bokeh.objects import Range1d

class bokehPlotInterface():
	def __init__(self, plot_mode="server"):
		if plot_mode != "server":
			print "request bokeh plot mode not set up yet",plot_mode
		else:
			output_server("plot.html")#, load_from_config=False)

		self.renderer = []
		self.ds = {}
		self._first_plot = True

	def init_plot(self, data, timestring):
		figure(x_range = Range1d(start=-0.5, end=3.5))
		hold()
	
		scatter(data['x'], y=data['y'], alpha=0.3, color=data['color'], size=data['size'])

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

        	cursession().store_objects(self.ds)
		print "try to update plot title",str(timestring)
		curplot().title = "Subway Visualization " + str(timestring)
	
	def plot(self, data, timestring):
		if self._first_plot:
			self.init_plot(data, timestring)
		else:
			self.animate_plot(data, timestring)

