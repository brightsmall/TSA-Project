# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
import re




working_directory = "D:\\TSA\\stage 1\\Predictions from 1200 Steps\\"


i=0

for file in os.listdir(working_directory):
    if file.endswith(".csv"):
        df = pd.read_csv(working_directory + file)
        df.columns = ['TMP','Probability','Id']
        file_string = str(file)
        split_string = file_string.split("_")
        zone_num = split_string[1]      
        df['Id'] = df['Id'] + "_Zone" + zone_num
        if i==0:
            output_df = df
        else:
            output_df=output_df.append(df)
        i+=1
       
output_df = output_df[['Id','Probability']]
      
output_df.to_csv(working_directory+'pred_1200_steps.csv',index=False)      
        