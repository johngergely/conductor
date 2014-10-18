import re
import time

def collect_strings():
    with open("bokeh.server.dat",'r') as f:
        text = f.read()
    docid_result = re.findall(r'docid\"\: \"[(\w\d\-)]*',text)
    id_result = re.findall(r'\"id\"\: \"[(\w\d\-)]*',text)
    docid = docid_result[0].split()[1][1:]
    id_string = id_result[0].split()[1][1:]
    return docid, id_string

def make_URL(SERVER_URL, embed_data):
    #docid, id_string = collect_strings()
    #URL = SERVER_URL + "bokeh/doc/" + docid + "/" + id_string
    write_template(embed_data)

def make_embed_script(embed_data):
    add_escapes = re.sub(r"\"", "\\\"", embed_data)
    formatted = ""
    for segment in add_escapes.split("\n"):
        formatted = formatted + segment + " "
    #print "embed data",embed_data
    #print "with escapes",add_escapes
    #print "concatenated",formatted
    with open("embed_bokeh_plot.js",'w') as f:
        f.write("document.write(\"" + formatted + "\")")

PLOT_PX = 600
OFFSET = int(0.5*PLOT_PX)
TOP_HEIGHT = 113
LEFT_WIDTH = 239
LEFT_OFFSET = LEFT_WIDTH + OFFSET
BOTTOM_OFFSET = TOP_HEIGHT + PLOT_PX + 50 #50 accounts for bokeh plot toolbar
BOTTOM_HEIGHT = 37
TOTAL_HEIGHT = PLOT_PX + TOP_HEIGHT + BOTTOM_OFFSET + BOTTOM_HEIGHT
BUTTON_WIDTH = 600

def write_template(embed_data):
    with open("index.html",'w') as f:
            f.write("<html>\n")
	    f.write("<title>CONDUCTOR - Real-time NYC Subway Visualization and Analytics</title>\n")
            f.write("<body style=\"font-family:helvetica;color:#4C4E52\">\n")

            f.write("<div id=\"outer-frame\" width=100%>\n\
	             <div id=\"reference\" style=\"position:absolute; left:50%; top:0px\">\n\
                <div id=\"left\" height=704px width=" + str(LEFT_WIDTH) + "px style=\"position:absolute; top:0px; left:-" + str(LEFT_OFFSET) + "px\"><img src=\"r32_left.jpg\" /></div>\n\
                    <div id=\"top\" height=" + str(TOP_HEIGHT) + "px width=" + str(PLOT_PX) + " style=\"position:absolute; top:-2px; left:-" + str(OFFSET) + "px;\"><img src=\"r32_top.jpg\" /></div>\n\
	    <div id=\"display\" width=" + str(PLOT_PX) + "px height=" + str(PLOT_PX) + "px style=\"position:absolute; top:114px; left:-" + str(OFFSET) + "px\">\n")
            f.write(embed_data)

            f.write("</div>\n<div id=\"bottom\" height=37px width=" + str(PLOT_PX)+ "px style=\"position:absolute; top:" + str(BOTTOM_OFFSET) + "px; left:-" + str(OFFSET) + "px\"><img src=\"r32_bottom.jpg\" /></div>\n\
                    <div id=\"right\" height=704px width=223px style=\"position:absolute; top:4px; left:" + str(OFFSET) + "px\"><img src=\"r32_right.jpg\" /></div>\n")
            f.write("<div id=\"button-area\" width=" + str(BUTTON_WIDTH) + " height=150px style=\"z-index:10; position:absolute; top:" + str(TOTAL_HEIGHT) + "; left:-" + str(0.5*BUTTON_WIDTH) + "\">")

            f.write("</div>\n<div id=\"text-area\" style=\"z-index:-10; margin:50px 50px 20px 50px; position:absolute; top:80%\">")
	    f.write("<h1>Using Conductor</h1>\n")
            f.write("<p>The interactive plot should be displayed here. Refresh the page if it is not displayed, or if you are not seeing the visualization update every few seconds.</p>\n")
            f.write("<p>You can select interactive tools along the right side of the plot: pan, box zoom, resize, and reset view.</p>\n")
            f.write("<p>This plot represents the real-time state of the 4/5/6 lines of the New York City subway system. By hovering over train and station objects, you can view data about the system, including geographical location, and trip duration and current status.</p>\n")
            f.write("<p>If you zoom in along the routes, you'll notice yellow shading along certain segments. This is a representation of how much trip times vary for the given segment. The width of the yelow-shaded area is proportional to the ratio of the median trip time along that semgent to the minimum trip time along that segment. In other words, for a segment of the trip with a wide shaded area, the time to cover this segment varies considerably and this segment is a source of delays.</p>\n")

            f.write("<br>")
	    f.write("<h1>About Conductor</h1>\n")
            f.write("<p>Conductor uses the New York City subway system's real-time data feed to do analysis and interactive visualization of the subways, incorporating live as well as historical data.</p>\n")


	    f.write("<p>Conductor is written in Python and makes use of the Bokeh plotting library. It is an open source project under active development. Here is the project on <a href=\"http://github.com/johngergely/conductor\">github</a>. You'll find the code repository, additional technical documentation, and in-process data analysis.</p>\n")

	    t_string = str(time.ctime())
            f.write("<p>This embed page was last generated: " + t_string + "</p>\n")

            f.write("</div>\n</div>\n</body>\n</html>")

if __name__=="__main__":
    make_URL("http://104.131.255.76:5006/", "\n\nNULLNULLNULL\n\n")
    print "Updated index.html"
