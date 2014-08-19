import re

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

def write_template(embed_data):
    with open("index.html",'w') as f:
            f.write("<html>\n")
	    f.write("<title>CONDUCTOR - Real-time NYC Subway Visualization and Analytics</title>\n")
            f.write("<body style=\"font-family:helvetica;color:#4C4E52\">\n")
            
            f.write(embed_data)

            f.write("<br>")
            f.write("<br>")
	    f.write("<h1>Using Conductor</h1>\n")
            f.write("<p>The interactive plot should be displayed here. Refresh the page if it is not displaying correctly.</p>\n")
            f.write("<p>You can select interactive tools along the right side of the plot: pan, box zoom, resize, and reset view.</p>\n")
            f.write("<p>This plot represents the real-time state of the 4/5/6 lines of the New York City subway system. By hovering your mouse over train and station objects, you can view data about the system, including geographical location, and trip duration and current status.</p>\n")
            f.write("<p>The yellow shading along the routes is a conceptual visualization of much trip times vary along the given segment. A width of the shaded area is proportional to the ratio of the median trip time along that semgent to the minimum trip time along that segment. In other words, a segment of the trip with a wide shaded area is a source of delays.</p>\n")

            f.write("<br>")
            f.write("<br>")
	    f.write("<h1>About Conductor</h1>\n")
            f.write("<p>Conductor uses the New York City subway system's real-time data feed to do analysis and interactive visualization of the subways, incorporating live as well as historical data.</p>\n")


	    f.write("<p>Conductor is written in Python and makes use of the Bokeh plotting library. It is an open source project under active development. Here is the project on <a href=\"http://github.com/johngergely/conductor\">github</a>.</p>\n")

            f.write("</body>\n")
            f.write("</html>\n")

if __name__=="__main__":
    make_URL("http://104.131.255.76:5006/")
    print "Updated index.html"
