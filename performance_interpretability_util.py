import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import torch
import numpy as np
from metapop1 import metapop1
import copy
import torch
import os
import pickle
import time

def load_episodes_human(envsetting):
    '''
    load files with the given environment setting and get list of all filenames and episode dicts.
    '''
    base_dir = './human_play_results/'
    pickle_filenames = []
    episodes = []
    for filename in os.listdir(base_dir):
        if f'_{envsetting}_' in filename and filename.endswith('.pkl'):
            pickle_filenames.append(os.path.join(base_dir, filename))
            # load file
            with open(os.path.join(base_dir, filename), "rb") as f:
                ep = pickle.load(f)
                episodes.append(ep)
    return pickle_filenames, episodes
    
def load_episodes_heuristic(envsetting, heuristic_setid):
    '''
    load files with the given environment setting and heuristic set ID and get list of all filenames and episode dicts.
    '''
    base_dir = './heuristics_play_results/'
    parseinfo_file = os.path.join(base_dir, 'parseinfo.txt')
    with open(parseinfo_file, 'r') as f:
        lines = f.readlines()

    # collect filenames belonging to the requested set
    target_set = f'set{heuristic_setid}\n'
    set_filenames = []
    in_set = False
    for line in lines:
        if line == target_set:
            in_set = True
        elif line.startswith('set'):
            in_set = False
        elif in_set and line.strip():
            set_filenames.append(line.strip())
    pickle_filenames = []
    episodes = []
    for filename in set_filenames:
        filepath = os.path.join(base_dir, filename)
        pickle_filenames.append(filepath)
        with open(filepath, 'rb') as f:
            ep = pickle.load(f)
            episodes.append(ep)
    return pickle_filenames, episodes

def calc_performance_gap(envsetting, actiontype = 'human', iterations = 1000, info = None):

    # setting parameters
    ## RL policy parameters
    if envsetting == 18:
        config = {'seed': 578396, 'paramset': 30}
    elif envsetting == 20:
        config = {'seed': 398845, 'paramset':32}
    elif envsetting == 21:
        config = {'seed':  252358, 'paramset': 34}
    else:
        raise ValueError(f"Invalid envsetting: {envsetting}.")

    # get list of episodes to work on
    if actiontype == 'human':
        pickle_filenames, episodes = load_episodes_human(envsetting)
    elif actiontype == 'heuristic':
        pickle_filenames, episodes = load_episodes_heuristic(envsetting, info['heuristic_setid'])
    
    # load the env and policy
    env = metapop1({'settingID': envsetting})


    wd = f'./PPO_results/good_ones/seed{config["seed"]}_paramid{config["paramset"]}'
    policy_filename = f"{wd}/bestPolicyNetwork_{env.envID}_par{env.paramsetID}_set{env.settingID}_PPO2.pt"
    rmsfilename = f"{wd}/bestPolicyrms_{env.envID}_par{env.paramsetID}_set{env.settingID}_PPO2.pkl"
    if os.path.exists(rmsfilename):
        standardize = True
        with open(rmsfilename, "rb") as f:
            rms = pickle.load(f)
    device = torch.device('cpu')  # Force CPU usage
    Policy = torch.load(policy_filename, weights_only=False)
    Policy.eval() # set to eval mode
    Policy = Policy.to(device) 
    gamma = 0.99
    fstack = 1 if not hasattr(Policy, 'fstack') else Policy.fstack
    print(f'number of episodes loaded: {len(episodes)}')
    for epi, ep in enumerate(episodes):
        # calculate L1 and euclidean distance between human and RL actions.
        ep['L1_a_dist_RLsampled_a'] = []
        ep['L1_a_dist_RLsampled_a_prob'] = []
        ep['L1_a_dist_RLsampled_a_prob_S'] = []
        ep['L1_a_dist_RLsampled_a_prob_R'] = []
        simsteps = np.arange(len(ep['envcheckpoints']))
        ep['performance_gap'] = []
        print(f'calculating performance gap for run {pickle_filenames[epi]} with {len(simsteps)} steps')
        for simstep in simsteps:
            ep['L1_a_dist_RLsampled_a'].append(np.abs(ep['actions'][simstep] - ep['RLactions'][simstep]).sum())
            # for calculating distance btw human action and RL priority scores 
            # (softmax for restoration, sigmoid for supplemenation) account for stop token sample in the human action.
            if env.kR < env.patchnum:
                if np.sum(ep['actions'][simstep][0:env.patchnum]) < env.kR:
                    stop_accounted_aR = np.concatenate((ep['actions'][simstep][0:env.patchnum], [1]))
                else:
                    stop_accounted_aR = np.concatenate((ep['actions'][simstep][0:env.patchnum], [0]))
            else:
                stop_accounted_aR = ep['actions'][simstep][0:env.patchnum]
            if env.kS < env.patchnum:
                if np.sum(ep['actions'][simstep][env.patchnum:]) < env.kS:
                    stop_accounted_aS = np.concatenate((ep['actions'][simstep][env.patchnum:], [1]))
                else:
                    stop_accounted_aS = np.concatenate((ep['actions'][simstep][env.patchnum:], [0]))
            else:
                stop_accounted_aS = ep['actions'][simstep][env.patchnum:]
            stop_accounted_action = np.concatenate((stop_accounted_aR,stop_accounted_aS))
            ep['L1_a_dist_RLsampled_a_prob'].append(np.abs(stop_accounted_action - ep['RLactions_prob'][simstep]).sum())
            rlen = env.patchnum if env.kR >= env.patchnum else env.patchnum + 1
            slen = env.patchnum if env.kS >= env.patchnum else env.patchnum + 1
            ep['L1_a_dist_RLsampled_a_prob_R'].append(np.abs(stop_accounted_aR - ep['RLactions_prob'][simstep][0:rlen]).sum())
            ep['L1_a_dist_RLsampled_a_prob_S'].append(np.abs(stop_accounted_aS - ep['RLactions_prob'][simstep][rlen:rlen+slen]).sum())
            # calculate performance gap
            env_og = ep['envcheckpoints'][simstep]
            V_human = 0
            V_pi = 0
            for i in range(2): # i=0 is human and i=1 is RL for first action.
                for _ in range(iterations):
                    envinstance = copy.deepcopy(env_og)
                    newstate = rms.normalize(envinstance.obs.copy()) if standardize else envinstance.obs.copy()
                    done = False
                    t = 0
                    rewards = 0
                    while done == False:
                        with torch.no_grad():
                            state = torch.tensor(newstate, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
                            action = Policy.getaction(state, get_action_only=True)
                            action = torch.squeeze(action).cpu().detach().numpy()
                        if t == 0:
                            if i == 0:
                                action = ep['actions'][simstep]
                            else:
                                action = ep['RLactions'][simstep]
                        _, reward, done, info = envinstance.step(action)
                        newstate  = rms.normalize(envinstance.obs.copy()) if standardize else envinstance.obs.copy()
                        rewards += reward*(gamma**t)
                        t += 1
                    if i == 0:
                        V_human += rewards
                    else:
                        V_pi += rewards
            # average across iterations
            V_human = V_human / iterations 
            V_pi = V_pi / iterations
            delV = V_human - V_pi
            ep['performance_gap'].append((delV, V_human, V_pi))
        # save the updated episodes with performance gap and action distance metrics
        # drop envcheckpoints
        ep.pop('envcheckpoints', None)
        ofilename = pickle_filenames[epi].replace(".pkl", "_perfgap_updated.pkl") # add perfgap_updated to filename
        # change directory from ./human_play_results/ to ./human_play_results/performance_gap_calculated/
        if actiontype == 'human':
            ofilename = ofilename.replace("./human_play_results/", "./human_play_results/performance_gap_calculated/")
        elif actiontype == 'heuristic':
            ofilename = ofilename.replace("./heuristics_play_results/", "./heuristics_play_results/performance_gap_calculated/")
        with open(ofilename, "wb") as f:
            pickle.dump(ep, f)
        print(f"Saved updated episode with performance gap to {ofilename}")            

if __name__ == "__main__":
    t0 = time.time()
    envsetting = 20 # 18(n5), 20(n10 median centrality), 21(n20)
    actiontype =  'heuristic'
    if actiontype == 'heuristic':
        info = {'heuristic_setid' : int(sys.argv[1])}
    else:
        info = None
    calc_performance_gap(envsetting, actiontype = actiontype, iterations = 1000, info = info)
    #calc_performance_gap(2) # for testing, use runid=1
    elapsed = time.time() - t0
    print(f"Done. Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")