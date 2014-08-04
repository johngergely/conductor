import gtfs_realtime_pb2
import urllib2
import httplib
import google.protobuf.message

# MTA Feed API Key
STREAM_URL = """http://datamine.mta.info/mta_esi.php?key="""
FEED_ID = """&feed_id=1"""
FEED_ID_2 = """&feed_id=2"""
# you need to obtain an API key from the MTA website and place it in this file in your local directory
with open("my_api_key",'r') as f:
    API_KEY = f.readline()[:-1]

def readMTADataStream(use_feed_id=FEED_ID):
    try:
        nyct_feed = gtfs_realtime_pb2.FeedMessage()
    except google.protobuf.message.DecodeError:
        print "protobuf DecodeError"
        return None
    #else:
    #    print "**************** Unexpected protobuf error ***********************"
    #    return None

    try:
        stream = urllib2.urlopen(STREAM_URL + API_KEY + use_feed_id)
        nyct_feed.ParseFromString(stream.read())
        stream.close()
        return nyct_feed
    except urllib2.URLError:
        print "Error: URLError - Failed to open URL."
        return None
    except httplib.IncompleteRead:
        print "Error: IncompleteRead - Failed to parse stream."
        return None
    else:
        print "************** Other failure ********************"
        return None

def processMTAData(nyct_feed, filename):
    timestamp = nyct_feed.header.timestamp
    with open(filename, 'a') as f:
        count = 0
        for entity in nyct_feed.entity:
            if entity.trip_update.trip.trip_id:
                stops = [stu for stu in entity.trip_update.stop_time_update]
                if len(stops)>0:
                    count += 1
                    f.write(str(timestamp) + ',' + str(entity.trip_update.trip.trip_id) \
		    + "," + str(entity.trip_update.trip.start_date) \
		    + ',' + str(stops[0].stop_id) + "," + str(stops[0].arrival.time) \
		    + "," + str(stops[0].departure.time) + '\n')
	print "dump complete",count

if __name__ == "__main__":
    import time
    max_time = 36000000
    local_time = 0
    wait_time = 30
    while(local_time < max_time):
        print local_time,
        attempts = 0
        nyct_feed = None
        while not nyct_feed:
            attempts += 1
            nyct_feed = readMTADataStream(use_feed_id=FEED_ID)
        print "(read successful",attempts,")",
        t = time.localtime()
        filename = "mta_data_v2." + str(t[0]) + "." + str(t[1]) + "." + str(t[2]) + ".csv"
        processMTAData(nyct_feed, filename)
        time.sleep(wait_time)
        local_time += wait_time
    print "data collect finished."
