# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:31:45 2021

@author: bgourdon
"""

import glassdoor_scraper1 as gs
import pandas  as pd

path = 'C:/Users/bgourdon/Documents/ds_salary_proj/chromedriver'

df = gs.get_jobs('data scientist', 15, False,path, 15 )