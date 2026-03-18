# parse all the heuristic play result files for the cluster.
import os
import numpy as np

# SETTING
## what to parse
envsetting = 23
idrange = np.arange(0,100)
## how many sets to make 
numsets = 10

# example filename: heuristic_play_metapop1_20_RLparam32_RLseed398845_20260316_175238_id2.pkl

# 1. get all heuristic play result files for the given envsetting and date range.
base_dir = './heuristics_play_results'
pickle_filenames = []
for filename in os.listdir(base_dir):
    if f'_{envsetting}_' in filename:
        id = int(filename.split('_id')[-1].split('.pkl')[0])
        if id in idrange:
            pickle_filenames.append(os.path.join(base_dir, filename))
            

# 2. divide the filenames into numset lists.
numfiles_perset = len(pickle_filenames) // numsets
file_set_list = []
for i in range(numsets):
    start_idx = i * numfiles_perset
    end_idx = (i + 1) * numfiles_perset if i < numsets - 1 else len(pickle_filenames)
    file_set_list.append(pickle_filenames[start_idx:end_idx])

# write a text file that goes like...
# set1
# filename1
# filename2
# set2
# filename3
# filename4
# ...

wd = './heuristics_play_results'
with open(f'{wd}/parseinfo.txt', 'w') as f:
    for i, file_set in enumerate(file_set_list):
        f.write(f'set{i+1}\n')
        for filename in file_set:
            f.write(f'{os.path.basename(filename)}\n')