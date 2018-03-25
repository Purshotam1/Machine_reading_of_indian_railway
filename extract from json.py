import json
from sortedcontainers import SortedDict
stationcode = {}
inverse = {}
with open('file.json') as json_data:
    dict = json.load(json_data)
    #print(dict["features"][0]["properties"]["code"])
    
    for i in range(len(dict["features"])):
        zone = dict["features"][i]["properties"]["zone"]
        no = dict["features"][i]["properties"]["number"]
        if no==None or zone == None:
            continue
        stationcode[no] = zone
stationcode = SortedDict(stationcode)
with open('train_no_to_zone.txt', 'w') as f:
    json.dump(stationcode,f)
