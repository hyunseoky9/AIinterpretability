import os 
import numpy as np
import pickle
import pandas as pd


def call_in_data(envid, with_valdiff=False, merge = False):
    '''
    call in play data from human play and return a list of dataframe.
    envid= environment id.
    '''
    if with_valdiff:
        data_dir = './human_play_results/performance_gap_calculated'
    else:
        data_dir='./human_play_results'
    data_list = []
    # get all filenames in human_play_results folder
    files = os.listdir(data_dir)
    filenames = []
    for filename in files:
        if f'human_play_metapop1_{envid}' in filename:
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

def call_in_heuristic_data(envid, heuristic_type=1, with_valdiff=False, merge = False):
    '''
    call in play data from heuristic play and return a list of dataframe.
    envid= environment id.
    '''
    if with_valdiff:
        data_dir = './heuristics_play_results/performance_gap_calculated'
    else:
        data_dir='./heuristics_play_results'
    data_list = []
    # get all filenames in heuristics_play_results folder
    files = os.listdir(data_dir)
    filenames = []
    for filename in files:
        if f'heuristic{heuristic_type}_play_metapop1_{envid}' in filename:
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