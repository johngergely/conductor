import pandas as pd
from bokeh.plotting import *
from bokeh.objects import Range1d, HoverTool, Glyph
from bokeh.embed import autoload_server
from collections import OrderedDict

import time
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dataEngine import unit_perp
from createLink import make_embed_script
from geography import  importShapefile, extract_silhouette

#SERVER_URL = """http://104.131.255.76:5006/"""
SERVER_URL = """http://127.0.0.1:5006/"""

# color palette
BRICK = "#800000"
BLUE = "#0066CC"
GOLDENROD = "#FFCC00"
LT_YELLOW = "#FFE066"
LATE_MAGENTA = "#FF0066"
GREEN = "#009933"
GRAY = "#4C4E52"

PLOT_WIDTH=600
PLOT_HEIGHT=600

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
    for i in lineData.index:
        xx = lineData.loc[i,'x']
        yy = lineData.loc[i,'y']

        xs.append(xx)
        ys.append(yy)
        alphas.append(1.0)
        colors.append(BLUE)
        widths.append(1.0)
    return multi_line(xs, ys, alpha=alphas, color=colors, line_width=widths, line_cap="round", line_join="round")

def _plotPatches(lineData):
    xs = []
    ys = []
    alphas = []
    colors = []
    widths = []
    x_scale = lineData.iloc[1,'x']-lineData.iloc[0,'x']
    #print "spatial scale",x_scale
    for i in lineData.index:
        xx = lineData.loc[i,'x']
        yy = lineData.loc[i,'y']
        #tmin = float(lineData['t_min'][i])
        #t50 = float(lineData['t_50pct'][i])
        #t75 = float(lineData['t_75pct'][i])

        #amplitude = (t50 + 0.001)/(tmin+ 0.001)
        amplitude = 8.0
        U = np.array((xx[1]-xx[0], yy[1]-yy[0]))
	if np.dot(U,U) == 0.:
            continue
        else:
            r0 =  np.array((xx[0], yy[0]))
            r1 =  np.array((xx[1], yy[1]))
            w = 0.5 * amplitude * 0.005 * abs(x_scale[0]) 
            x_set, y_set =  _make_polygon(r0, r1, 0.2, w)

            xs.append(x_set)
            ys.append(y_set)
            alphas.append([0.])
            colors.append([GOLDENROD])

    lineDataSource = ColumnDataSource(data={'alpha':alphas})
    return patches(xs, ys, fill_color=colors, fill_alpha=alphas, line_alpha=0.0, line_color=GOLDENROD, source=lineDataSource)

def _plotGeography(downsample_interval=1):
        mh_index = -3
        bx_index = -1
        bk_index = -1
        qu_index = -1
        shp_source = "data/nybb_14b_av/nybb.shp"

        shapes = importShapefile(shp_source)

        shapes_x_lists = []
        shapes_y_lists = []

        for shp_i,geo_i in [(1,mh_index), (2, bx_index), (3,bk_index), (4,qu_index)]:
                #print "SHAPEFILE SHAPE",shp_i,geo_i
                xx,yy = extract_silhouette(shapes[shp_i], geo_i)
                ## adapted from http://stackoverflow.com/questions/15178146/line-smoothing-algorithm-in-python
                N_resample = len(xx)/downsample_interval
                #print "N_resample",N_resample
                t = np.linspace(0, 1, len(xx))
                t2 = np.linspace(0, 1, N_resample)
                #print "lengths xx t t2",len(xx),len(t),len(t2)
                x2 = np.interp(t2, t, xx)
                y2 = np.interp(t2, t, yy)
                sigma = 1.2
                #x3 = gaussian_filter1d(x2, sigma)
                #y3 = gaussian_filter1d(y2, sigma)
                xx_resampled = gaussian_filter1d(x2, sigma)
                yy_resampled = gaussian_filter1d(y2, sigma)

                #xx_resampled = np.interp(t, t2, x3)
                #yy_resampled = np.interp(t, t2, y3)
                #print "lengths xx t t2 xx_resample",len(xx),len(t),len(t2),len(xx_resampled)
                ##xx_resampled = [xx[i] for i in range(0,len(xx),downsample_interval)]
                ##yy_resampled = [yy[i] for i in range(0,len(yy),downsample_interval)]
                ######################
                #print "resampled shapefiles have length", len(xx_resampled), len(yy_resampled)
                shapes_x_lists.append(xx_resampled)
                shapes_y_lists.append(yy_resampled)
        return multi_line(xs=shapes_x_lists, ys=shapes_y_lists, alpha=0.45, line_width=5, color=GRAY)

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
                        self.TOOLS = ['pan', 'box_zoom', 'wheel_zoom', 'reset', 'hover']
                else:                                                          
                        self.TOOLS = ['pan', 'box_zoom', 'wheel_zoom', 'reset'] 

		#choosefields = {'train':['name','approaching','duration','late'], 'station':['name','x','y']}
		columnDict = {}
		hoverlist = []
		for f in fields:
			columnDict[f] = data[f].values
			#hoverlist.append(("\"" + f + "\"" , "\"@" + f + "\""))
                self.source = ColumnDataSource(data=columnDict)
		self.hoverDict = OrderedDict(hoverlist)

	def _init_plot(self, scatterData, lineData, hoverFields, timestring):
                self._init_hover(scatterData, hoverFields)
		#staticplot = self.static_plot(staticData)

                #self.xmin = data['x'].min()
                #self.ymax = data['y'].max()

                figure(x_range=Range1d(start=self.xmin, end=self.xmax),
			y_range=Range1d(start=self.ymin, end=self.ymax),
		        title = "",#CONDUCTOR - Real-time Subway Visualization and Analytics",
		        title_text_font_size = "14pt",
		        title_text_color = GRAY, 
			x_axis_type=None,
			y_axis_type=None,
                       	min_border=0,
                       	outline_line_color=None,
                        plot_width=PLOT_WIDTH,
                        plot_height=PLOT_HEIGHT)

		hold()

		scatter(scatterData['x'], y=scatterData['y'], alpha=scatterData['alpha'], color=scatterData['color'], size=scatterData['size'], source=self.source, tools=self.TOOLS)
		#circle(x=scatterData['x'], y=scatterData['y'], alpha=scatterData['alpha'], color=scatterData['color'], size=scatterData['size'], source=self.source, tools=self.TOOLS)
               
                # get hold of this to display hover data
                #hover = [tools for tools in curplot().tools if isinstance(tools, HoverTool)][0] ## outdated for bokeh version < 0.6
                self.hover = curplot().select(dict(type=HoverTool)) # bokeh version >= 0.6
		#self.hover.tooltips = self.hoverDict
                self.hover.useString = True
                self.hover.styleProperties = {"color":"white", "backgroundColor":GRAY, "z-index":"10"}
                #self.hover.stringData = list(scatterData['formatted_string'])

                _plotGeography(downsample_interval=39)
                #_plotPatches(lineData)
                _plotLineData(lineData)

                text([995864], [191626], text="Brooklyn", text_baseline="middle", text_align="left",   text_font_size="14", text_color=GRAY, text_font="helvetica", angle=0)
                text([992549], [236269], text="Manhattan", text_baseline="middle", text_align="right", text_font_size="14", text_color=GRAY, text_font="helvetica", angle=0)
                text([1018884], [236646], text="The Bronx", text_baseline="middle", text_align="left", text_font_size="14", text_color=GRAY, text_font="helvetica", angle=0)
                text([1000884], [213246], text="Queens", text_baseline="middle", text_align="left", text_font_size="14", text_color=GRAY, text_font="helvetica", angle=0)

                self.curplot = curplot
                #self.curplot.plot_height = PLOT_HEIGHT
                #self.curplot.plot_width = PLOT_WIDTH

		xaxis().grid_line_color = None
		yaxis().grid_line_color = None
		show()

                # get hold of this to refresh data for animation
		self.renderers = [r for r in curplot().renderers if isinstance(r, Glyph)]
                #print "RENDERERS"
                #for rend in self.renderers:
                #    print rend,rend.data_source.data.keys()
		#self.renderer = [r for r in curplot().renderers if isinstance(r, Glyph)][0]
		#self.ds = self.renderer.data_source

		self._first_plot = False

        #@profile
	def _animate_plot(self, data, lineData, fields):
                for f in self.renderers[0].data_source.data.keys():
                        if not f in data.columns:
                                if f=='line_color' or f=='fill_color':
			                self.renderers[0].data_source.data[f] = data['color']
                                elif f=='line_alpha' or f=='fill_alpha':
			                self.renderers[0].data_source.data[f] = data['alpha']
                                #else:
                                #        print "KEY",f,"NOT FOUND IN",data.keys()
                        else:
			        self.renderers[0].data_source.data[f] = data[f]
		self.renderers[0].data_source.data["formatted_string"] = data.get("formatted_string")
        	cursession().store_objects(self.renderers[0].data_source)
                ###print "setting alphas"
                ###print lineData['alpha']
                ##patchRendererID = 2
                ##self.renderers[patchRendererID].data_source.data['alpha'] = lineData['alpha']
                ##self.renderers[patchRendererID].data_source.data['fill_alpha'] = lineData['alpha']
        	##cursession().store_objects(self.renderers[patchRendererID].data_source)

	def plot(self, scatterData, lineData, dynamicFields, hoverFields, timestring):
		if self._first_plot:
			self._init_plot(scatterData, lineData, hoverFields, timestring)
		else:
			self._animate_plot(scatterData, lineData, dynamicFields)

                EMBED_DATA = autoload_server(curplot(), cursession())
                #make_URL(SERVER_URL, EMBED_DATA)
                make_embed_script(EMBED_DATA)

class dummyPlotInterface(bokehPlotInterface):
        def __init__(self):
                print "dummy plot manager initialized; no plots will be generated"

	def plot(self, scatterData, lineData, dynamicFields, hoverFields, timestring):
                print "PLOT DATA"
                print timestring
                print scatterData.shape
                print lineData.shape
                
if __name__=="__main__":
	print "TESTING PLOT WITH DATA SET"
        #output_server("whatever.html", load_from_config=False)#test_plot_functionality.html")
        output_server("whatever.html")#test_plot_functionality.html")
        #output_file("whatever.html")#test_plot_functionality.html")
        #cursession().load_from_config = False
        TOOLS = ['pan', 'box_zoom', 'wheel_zoom', 'resize', 'reset', 'hover']
        data = pd.DataFrame(columns=['x','y','name','type','records','iter','whatyouknow'], 
                            data=[[1.,1.,'a','character',"vinyl",0,None],
                                  [2.,1.,'fox','animal',"yellow<br>blue<br>gray",0,'octopus']])#,
                                  #[0.,0.,'jimbo','guy',"man i am sick of it",10,'octopus']])
        fields = ['name','type','iter','records','whatyouknow']
        columnDict = {}
        hoverlist = []
        for f in fields:
                columnDict[f] = data.get(f)
                hoverlist.append((f, "@" + f))
        #print "test hoverlist",hoverlist
        source = ColumnDataSource(data=columnDict)
        hoverDict = OrderedDict(hoverlist)
        #print "HOVER FIELDS"
        #print hoverDict
        #for k in source.data.keys():
        #        print k,source.data[k]
        figure(x_range=Range1d(start=-0.25, end=2.25),
	    y_range=Range1d(start=-0.25, end=1.25))
        hold()

        scatter(data['x'], y=data['y'], size=10, source=source, tools=TOOLS)
        circle(x=[0.], y=[0.], size=20, color=["rgba(255,0,0,1)"], source=ColumnDataSource(data={'x':[0.], 'y':[0.], 'color':["rgba(255,0,0,1)"]}), tools=TOOLS)

        # get hold of this to display hover data
        hover = curplot().select(dict(type=HoverTool)) # bokeh version >= 0.6
        hover.tooltips = hoverDict
        hover.useString = False#True
        #hover.stringData = ["sending as a string<br> even w a new line.", "second guy<br>&nbsp&nbspindent"]
        hover.stringStyle = {"color":"white", "backgroundColor":"rgba(0,0,255,0.4)"}

        show()

        #get hold of this to refresh data for animation
        renderer_list = [r for r in curplot().renderers if isinstance(r, Glyph)]
        #print "renderers",renderer_list
        #for eend in renderer_list:
        #    print rend.data_source.data.keys()
        renderer = renderer_list[0]
        ds = renderer.data_source
        for t in range(20):
            ys = [y-t*0.03 for y in data['y']]
            xs_1 = [t*0.03]
            #ds.data['y'] = ys
            renderer_list[0].data_source.data['y'] = ys
            cursession().store_objects(renderer_list[0].data_source)
            renderer_list[1].data_source.data['x'] = xs_1#"rgba(255," + str(10*t) + ", 0,1)"
            renderer_list[1].data_source.data['color'] = ["rgba(255," + str(10*t) + ", 0,1)"]
            renderer_list[1].data_source.data['fill_color'] = ["rgba(255," + str(10*t) + ", 0,1)"]
            renderer_list[1].data_source.data['line_color'] = ["rgba(255," + str(10*t) + ", 0,1)"]
            cursession().store_objects(renderer_list[1].data_source)
            #cursession().store_objects(ds)
