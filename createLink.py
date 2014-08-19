import re

def collect_strings():
    with open("bokeh.server.dat",'r') as f:
        text = f.read()
    docid_result = re.findall(r'docid\"\: \"[(\w\d\-)]*',text)
    id_result = re.findall(r'\"id\"\: \"[(\w\d\-)]*',text)
    docid = docid_result[0].split()[1][1:]
    id_string = id_result[0].split()[1][1:]
    return docid, id_string

def make_URL(SERVER_URL):
    docid, id_string = collect_strings()
    URL = SERVER_URL + "bokeh/doc/" + docid + "/" + id_string
    write_template(URL)

def write_template(URL):
    with open("index.html",'w') as f:
            f.write("<html>\n")
	    f.write("<title>CONDUCTOR - Real-time NYC Subway Visualization and Analytics</title>\n")
            f.write("<body>\n")

	    f.write("<p>Conductor uses the New York City subway system's real-time data feed to do analysis and interactive visualization of the subways, incorporating live as well as historical data.</p>\n")

            f.write("<p><a href=\"" + URL + "\">View the real-time visualization.</a></p>\n")

	    f.write("<p>Conductor is written in Python. It is an open source project under active development. Here is the project on <a href=\"http://github.com/johngergely/conductor\">github</a>.</p>\n")

            f.write("</body>\n")
            f.write("</html>\n")

if __name__=="__main__":
    make_URL("http://104.131.255.76:5006/")
    print "Updated index.html"
