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
            f.write("<html>")
            f.write("<body>")
            f.write("<a href=URL>View real-time visualization.</a>")
            f.write("</body>")
            f.write("</html>")
