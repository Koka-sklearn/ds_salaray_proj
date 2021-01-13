# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:57:17 2021

@author: bgourdon
"""

# Libraries needed for the tutorial

import pandas as pd
import requests
import io
    
# Downloading the csv file from your GitHub account

url = "https://raw.githubusercontent.com/PlayingNumbers/ds_salary_proj/master/glassdoor_jobs.csv" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content

# Reading the downloaded content and turning it into a pandas dataframe

df = pd.read_csv(io.StringIO(download.decode('utf-8')))

# Printing out the first 5 rows of the dataframe

#print (df.head(15))

#salary parsing

 
    
   
    
    #remove data -1 in salary
df = df[df['Salary Estimate'] != '-1']
    
    #create salary function to start from 0 to first parenthese
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])

    #from salary function to replace k and $ by blanks
minus_kd = salary.apply(lambda x: x.replace ('K','').replace('$',''))

    #create a column where per hour is in and check 1 in or else 0
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'Employer Provided Salary' in x.lower() else 0)

    #use lower to minuscule la casse
min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

#create 3 columns min and max salary before and after the '-' THEN average
df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

#Company name text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

#state field
    # 0 = first enry 1 = second, etc...
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.job_state.value_counts()

    #print(df.job_state.value_counts())
    #LOS ANGELES A CHeck

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

#age of Company
#scale between year and age fo the company
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2020 - x)

 #parsing of job description (python, etc.)

#top data science tools

#python
df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
#print(df.python.value_counts())
#R studio
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
#print(df.R_yn.value_counts())
#spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
#print(df.spark_yn.value_counts())
#aws
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
#print(df.aws_yn.value_counts())
#excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
#print(df.excel_yn.value_counts())

df.columns
df_out = df.drop(['Unnamed: 0'], axis = 1)

df_out.to_csv('salary_data_cleaned.csv',index= False)

pd.read_csv('salary_data_cleaned.csv')





























