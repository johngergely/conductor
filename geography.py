from fiona import collection
from bokeh.plotting import *
from bokeh.objects import Range1d

def importShapefile(fname):
    with collection(fname, 'r') as c:
        shapes = [shape for shape in c]
    return shapes
    
def extract_silhouette(shp, index):
        xx = [c[0] for c in shp['geometry']['coordinates'][index][0]]
        yy = [c[1] for c in shp['geometry']['coordinates'][index][0]]
        return xx,yy

def plot_silhouette(xx,yy):
        figure()
        line(x=xx,y=yy)
        show()

if __name__=="__main__":

        xmin = 975672.769948
        xmax = 1051864.77272
        ymin = 149176.556361
        ymax = 268405.335508

        mh_index = -3
        bx_index = -1
        bk_index = -1
        shp_source = "data/nybb_14b_av/nybb.shp"

        shapes = importShapefile(shp_source)

        output_server("test.html")
        
        figure(x_range=Range1d(start=xmin, end=xmax), y_range=Range1d(start=ymin, end=ymax))
        hold()

        for shp_i,geo_i in [(1,mh_index), (2, bx_index), (3,bk_index)]:
                xx,yy = extract_silhouette(shapes[shp_i], geo_i)
                line(x=xx, y=yy)

        show()
