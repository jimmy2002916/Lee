#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:28:40 2018

@author: jimmyhomefolder
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import urllib
from urllib.request import urlopen
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("student_data.csv")

#%%   
def getGeoForAddress(address):
    try:
        addressUrl = "http://maps.googleapis.com/maps/api/geocode/json?address=" + address
        addressUrlQuote = urllib.parse.quote(addressUrl, ':?=/')
        response = urlopen(addressUrlQuote).read().decode('utf-8')
        responseJson = json.loads(response)
        lat = responseJson.get('results')[0]['geometry']['location']['lat']
        lng = responseJson.get('results')[0]['geometry']['location']['lng']
        print(address + '的经纬度是: %f, %f'  %(lat, lng))
        df.iloc[i, -1] = lat
        df.iloc[i, -2] = lng
    except:
        pass
 
if __name__ == '__main__':
    for i in range(0,100):
        address = df.iloc[i,2]
        getGeoForAddress(address)
    
df.to_csv("student_data_with_geo_info.csv")