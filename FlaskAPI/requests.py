# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:07:05 2021

@author: bgourdon
"""

import requests
from data_input import data_in
#api-endpoints
URL = 'http://127.0.0.1:5000/predict'
#params
headers = {"Content-Type": "application/json" }
#dictionnary in json
data = {"input":data_in}

r = requests.get(URL, headers=headers, json=data)

r.json()