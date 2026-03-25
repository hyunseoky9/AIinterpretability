"""
Heuristic Policy for Metapopulation Conservation MDP
=====================================================

A parameterized rule-based policy derived from analyzing PPO-trained RL agent behavior
across 11 episodes on a 10-patch median-centrality landscape.

The policy is designed to be scenario-agnostic: parameters can be tuned per scenario
(small/medium/large landscape, low/median/high centrality) via grid search or manual
evaluation.

Key RL behavioral patterns distilled into rules:
1. SUPPLEMENTATION is the primary tool for maintaining occupancy.
   - Almost exclusively targets unoccupied patches (X=0).
   - Strongly prefers patches with good habitat (H=1) over degraded (H=0).
   - Skips high-connectivity H=1 patches when system occupancy is high 
     (they can self-colonize).
   - Scales effort: supplements most/all extinct patches when few are extinct,
     but reduces effort when too many are extinct (cost-saving / triage).

2. RESTORATION is secondary and budget-constrained (max k_R per step).
   - Only targets degraded patches (H=0), never H=1.
   - Prioritizes patches by a score combining occupancy and connectivity:
     occupied high-C patches get restored first.
   - Uses full budget when degradation is moderate; gives up when degradation 
     is extreme (n_degraded >= ~7).

3. GIVE-UP BEHAVIOR: When the system is in severe decline (very high degradation),
   the RL stops acting entirely to avoid wasting cost on a lost cause.

Usage
-----
    policy = HeuristicPolicy(N=10, params=default_params_medium)
    actions = policy.act(observation)  # observation shape: (N, 4) with [X, H, C, t]
    
    # actions is np.array of shape (2*N,): first N = restoration, last N = supplementation
"""

import numpy as np


# ============================================================================
# DEFAULT PARAMETERS (tuned for 10-patch median-centrality scenario)
# ============================================================================




class HeuristicPolicy:
    """
    Parameterized heuristic policy for metapopulation conservation MDP.
    
    Parameters
    ----------
    N : int
        Number of habitat patches.
    params : dict
        Dictionary of tunable parameters (see default_params_medium for keys).
    """
    
    def __init__(self, N, params=None):
        default_params_medium = {
            # --- Supplementation parameters ---
            
            # Habitat quality gate: only supplement X=0 patches with H >= this value.
            # RL almost never supplements H=0 patches when occupancy is moderate.
            # Set to 1.0 to require good habitat; set to 0.0 to allow supplementing degraded patches too.
            "supp_H_threshold": 1.0,
            
            # Connectivity threshold for skipping supplementation on H=1 extinct patches.
            # When system occupancy is high (above supp_occ_comfort), patches with 
            # connectivity above this threshold are expected to self-colonize and are skipped.
            # From RL data: skipping ramps up around C ~ 2.0.
            "supp_skip_C_threshold": 2.0,
            
            # System occupancy level above which the RL starts skipping high-C patches.
            # Below this, it supplements almost all H=1 extinct patches.
            "supp_occ_comfort": 0.7,
            
            # When supplementing H=0 extinct patches (if supp_H_threshold < 1),
            # this is the minimum connectivity required.
            "supp_H0_min_C": 1.5,
            
            # Maximum fraction of extinct patches to supplement in a single timestep.
            # The RL doesn't always supplement all extinct patches (cost management).
            # Set to 1.0 to supplement all eligible, or lower to cap spending.
            "supp_max_fraction": 1.0,
            
            # Give-up threshold: if the fraction of degraded patches exceeds this,
            # stop supplementing entirely (the system is collapsing).
            "supp_giveup_degraded_frac": 0.7,
            
            # --- Restoration parameters ---
            
            # Maximum number of patches to restore per timestep (scenario-specific).
            "rest_budget": 2,
            
            # Scoring weights for prioritizing which degraded patches to restore.
            # Score = w_X * X + w_C * C_normalized
            # RL clearly prefers occupied patches (X=1) and high-connectivity patches.
            "rest_w_occupancy": 1.0,   # weight for occupancy (X=1 vs X=0)
            "rest_w_connectivity": 1.0, # weight for normalized connectivity
            
            # Minimum system occupancy fraction to bother restoring.
            # When occupancy is very low, RL gives up on restoration.
            "rest_min_occ_frac": 0.2,
            
            # Minimum system habitat fraction to bother restoring.
            # When too much is degraded, RL stops restoring (futile).
            "rest_giveup_degraded_frac": 0.7,
        }


        # Scenario-specific starting points (to be tuned by evaluation)
        default_params_small = {
            **default_params_medium,
            "rest_budget": 1,
        }

        default_params_large = {
            **default_params_medium,
            "rest_budget": 4,
        }
        
        self.N = N
        self.params = params if params is not None else default_params_medium.copy()
    
    def act(self, observation):
        """
        Select conservation actions given current system state.
        
        Parameters
        ----------
        observation : np.ndarray, shape (N, 4)
            Per-patch state: [X_i, H_i, C_i, t] where
            X = occupancy (0/1), H = habitat quality (0/1),
            C = connectivity (float), t = timestep.
        
        Returns
        -------
        actions : np.ndarray, shape (2*N,)
            Binary action vector. First N entries = restoration actions,
            last N entries = supplementation actions.
        """
        p = self.params
        N = self.N
        
        X = observation[:, 0]  # occupancy
        H = observation[:, 1]  # habitat quality
        C = observation[:, 2]  # connectivity
        
        actions = np.zeros(2 * N)
        
        occ_frac = X.mean()
        degraded_frac = 1.0 - H.mean()
        
        # =====================================================================
        # SUPPLEMENTATION DECISIONS (actions[N:2N])
        # =====================================================================
        
        # Give-up check: if degradation is extreme, stop all supplementation
        if degraded_frac < p["supp_giveup_degraded_frac"]:
            
            # Find all unoccupied patches
            extinct_idx = np.where(X == 0)[0]
            
            if len(extinct_idx) > 0:
                # Score each extinct patch for supplementation priority
                supp_scores = np.full(N, -np.inf)
                
                for i in extinct_idx:
                    if H[i] >= p["supp_H_threshold"]:
                        # H=1 extinct patch: high priority
                        # But skip if high-C and system is comfortable
                        if occ_frac >= p["supp_occ_comfort"] and C[i] >= p["supp_skip_C_threshold"]:
                            supp_scores[i] = -1.0  # low priority (expected to self-colonize)
                        else:
                            # Priority: lower C patches need help more
                            supp_scores[i] = 10.0 - C[i]
                    else:
                        # H=0 extinct patch: low priority, only if C is high enough
                        if C[i] >= p["supp_H0_min_C"]:
                            supp_scores[i] = 0.0  # base low priority
                        # else: don't supplement (score stays -inf)
                
                # Select top patches up to budget
                max_supp = max(1, int(p["supp_max_fraction"] * len(extinct_idx)))
                eligible = np.where(supp_scores > -np.inf)[0]
                
                if len(eligible) > 0:
                    # Sort by score descending
                    ranked = eligible[np.argsort(-supp_scores[eligible])]
                    selected = ranked[:max_supp]
                    actions[N + selected] = 1.0
        
        # =====================================================================
        # RESTORATION DECISIONS (actions[0:N])
        # =====================================================================
        
        # Give-up check: if degradation is extreme or occupancy too low, skip
        if (degraded_frac < p["rest_giveup_degraded_frac"] and 
            occ_frac >= p["rest_min_occ_frac"]):
            
            degraded_idx = np.where(H == 0)[0]
            
            if len(degraded_idx) > 0:
                # Score each degraded patch for restoration priority
                # Normalize connectivity to [0, 1] range for scoring
                C_max = C.max() if C.max() > 0 else 1.0
                C_norm = C / C_max
                
                rest_scores = np.full(N, -np.inf)
                for i in degraded_idx:
                    rest_scores[i] = (p["rest_w_occupancy"] * X[i] + 
                                      p["rest_w_connectivity"] * C_norm[i])
                
                # Select top patches up to restoration budget
                eligible = np.where(rest_scores > -np.inf)[0]
                if len(eligible) > 0:
                    ranked = eligible[np.argsort(-rest_scores[eligible])]
                    budget = min(p["rest_budget"], len(ranked))
                    selected = ranked[:budget]
                    actions[selected] = 1.0
        
        return actions
    
    def __repr__(self):
        return f"HeuristicPolicy(N={self.N}, params={self.params})"


# ============================================================================
# HELPER: Parameter grid for tuning
# ============================================================================

def get_param_grid():
    """
    Returns a dictionary of parameter names -> candidate values for grid search.
    
    You can evaluate each combination on your environment and pick the best
    per scenario. Start with the coarse grid below and refine.
    """
    return {
        "supp_H_threshold": [0.0, 1.0],
        "supp_skip_C_threshold": [1.5, 2.0, 2.5, 999.0],  # 999 = never skip
        "supp_occ_comfort": [0.6, 0.7, 0.8, 1.0],          # 1.0 = never skip
        "supp_H0_min_C": [0.0, 1.0, 1.5, 2.0],
        "supp_max_fraction": [0.5, 0.75, 1.0],
        "supp_giveup_degraded_frac": [0.5, 0.6, 0.7, 0.8, 1.0],  # 1.0 = never give up
        "rest_w_occupancy": [0.0, 0.5, 1.0, 2.0],
        "rest_w_connectivity": [0.0, 0.5, 1.0, 2.0],
        "rest_min_occ_frac": [0.0, 0.1, 0.2, 0.3],
        "rest_giveup_degraded_frac": [0.5, 0.6, 0.7, 0.8, 1.0],
    }


def get_focused_param_grid():
    """
    A smaller, focused grid around the RL-derived defaults for faster tuning.
    Recommended for initial evaluation.
    """
    return {
        "supp_skip_C_threshold": [1.8, 2.0, 2.2, 999.0],
        "supp_occ_comfort": [0.6, 0.7, 0.8],
        "supp_giveup_degraded_frac": [0.6, 0.7, 0.8, 1.0],
        "rest_w_occupancy": [0.5, 1.0, 2.0],
        "rest_w_connectivity": [0.5, 1.0, 2.0],
        "rest_giveup_degraded_frac": [0.6, 0.7, 0.8, 1.0],
    }


# ============================================================================
# DEMO / VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Quick demo with a synthetic observation
    N = 10
    policy = HeuristicPolicy(N=N)
    
    # Simulate a state: 7 occupied, 3 extinct; 8 good habitat, 2 degraded
    obs = np.array([
        [1, 1, 1.5, 5],  # patch 0: occupied, good habitat
        [1, 1, 1.2, 5],  # patch 1: occupied, good habitat
        [0, 1, 0.8, 5],  # patch 2: extinct, good habitat -> supplement
        [1, 0, 1.8, 5],  # patch 3: occupied, degraded -> restore (high priority)
        [1, 1, 2.1, 5],  # patch 4: occupied, good habitat
        [0, 0, 1.2, 5],  # patch 5: extinct, degraded -> maybe supplement
        [1, 1, 1.7, 5],  # patch 6: occupied, good habitat
        [0, 1, 2.3, 5],  # patch 7: extinct, good habitat, high C -> maybe skip supp
        [1, 1, 1.9, 5],  # patch 8: occupied, good habitat
        [1, 0, 0.5, 5],  # patch 9: occupied, degraded -> restore (lower priority)
    ])
    
    actions = policy.act(obs)
    
    print("Demo observation:")
    print(f"  Occupancy:    {obs[:, 0].astype(int)}")
    print(f"  Habitat:      {obs[:, 1].astype(int)}")
    print(f"  Connectivity: {obs[:, 2].round(2)}")
    print(f"\nHeuristic actions:")
    print(f"  Restoration:     {actions[:N].astype(int)}  (patches: {np.where(actions[:N])[0]})")
    print(f"  Supplementation: {actions[N:].astype(int)}  (patches: {np.where(actions[N:])[0]})")
    print(f"\nExpected behavior:")
    print(f"  Restore: patch 3 (occupied, degraded, high C) and patch 9 (occupied, degraded)")
    print(f"  Supplement: patch 2 (extinct, H=1, low C) and patch 7 depends on occ_comfort")
    print(f"  System occ_frac = {obs[:,0].mean():.1f}, degraded_frac = {1-obs[:,1].mean():.1f}")
