"""
    This is used to open the data within the CSV files
    https://www.kaggle.com/puxama/bostoncsv/tasks
    https://www.kaggle.com/c/boston-housing
"""
import pandas as pd
inputs = pd.read_csv("Data/Boston.csv", usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13])
inputs_array = inputs.to_numpy(dtype='float32')

targets = pd.read_csv("Data/Boston.csv", usecols=[14])
targets_array = targets.to_numpy(dtype='float32')

