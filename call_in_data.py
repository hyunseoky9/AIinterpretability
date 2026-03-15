import os 
import numpy as np
import pickle
import pandas as pd


def call_in_data(n, with_valdiff=False, merge = False):
    '''
    call in play data from human play and return a list of dataframe.
    n= patch num.
    '''
    if with_valdiff:
        data_dir = './human_play_results/performance_gap_calculated'
    else:
        data_dir='./human_play_results'
    data_list = []
    # get all filenames in human_play_results folder
    files = os.listdir(data_dir)
    matchingenvid = {20:21, 10:20, 5:18} # key -> env id.
    filenames = []
    for filename in files:
        if f'human_play_metapop1_{matchingenvid[n]}' in filename:
            with open(os.path.join(data_dir, filename), 'rb') as f:
                data = pickle.load(f)
            data_list.append(data)
            filenames.append(filename)
    
    if merge:
        merged_data = {}
        for key in data_list[0].keys():
            merged_data[key] = []
            for data in data_list:
                if type(data[key]) == list:
                    merged_data[key].extend(data[key])
                else:
                    merged_data[key].append(data[key])
        return merged_data, filenames
    
    return data_list, filenames

