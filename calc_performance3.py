from call_paramset import call_env
import numpy as np
import torch

def calc_performance(env, device, rms, fstack, policy, episodenum=1000, t_maxstep = 1000, deterministic_eval=False):
    """
    same as calc_performance.py but built for PPO algorithms. 
    non-parallelized version.
    calculate the performance of the agent in the environment.
    """
    print('serial calc_performance called')
    avgrewards = 0

    for i in range(episodenum):
        rewards = 0
        env.reset()
        stack = np.concatenate([rms.normalize(env.obs)]*fstack) if rms is not None else np.concatenate([env.obs]*fstack)
        done = False
        t = 0
        while done == False:
            state = rms.normalize(env.obs) if rms is not None else env.obs
            stack = np.concatenate((stack[len(state):], state))

            with torch.no_grad():                                   # <– no grads here
                s = torch.as_tensor(stack.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                if deterministic_eval:
                    action = policy.get_deterministic_action(s)
                else:
                    action = policy.getaction(s, get_action_only=True)
                action = action.cpu().numpy().squeeze(0)
            _, reward, done, _ = env.step(action)
            rewards += reward
            if t >= (t_maxstep - 1):
                done = True
            t += 1
        avgrewards += rewards   
    return avgrewards/episodenum