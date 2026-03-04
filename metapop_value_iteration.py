import numpy as np
from itertools import combinations
from typing import Callable, Dict, Tuple, List

def _popcount(x: int) -> int:
    return x.bit_count()

def _bits_to_vec(mask: int, n: int) -> np.ndarray:
    # bit i corresponds to patch i
    return np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=np.int8, count=n)

def _all_subset_masks_upto(n: int, k: int) -> List[int]:
    """All subset bitmasks of {0..n-1} with size <= k."""
    masks = [0]
    for r in range(1, k + 1):
        for comb in combinations(range(n), r):
            m = 0
            for i in comb:
                m |= (1 << i)
            masks.append(m)
    return masks

def _row_normalize(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = P.astype(float).copy()
    rs = P.sum(axis=1, keepdims=True)
    bad = np.where(np.abs(rs.squeeze() - 1.0) > 1e-6)[0]
    if bad.size > 0:
        # Normalize rows; if a row sums to 0, make it uniform.
        for i in range(P.shape[0]):
            s = rs[i, 0]
            if s < eps:
                P[i, :] = 1.0 / P.shape[1]
            else:
                P[i, :] /= s
        print(f"[WARN] Ztrans rows did not sum to 1; row-normalized them. Offending rows: {bad.tolist()}")
    return P

def build_optimal_controller_fully_observable(
    env,
    kR: int = 1,
    kS: int = 1,
    include_do_nothing: bool = True,
    prob_prune: float = 1e-6,
) -> Dict[str, object]:
    """
    Finite-horizon DP for metapop1 fully observable MDP (no survey).
    Returns a dict with:
      - 'act': callable(state_vec)-> action_vec (uses t from state_vec)
      - 'act_from_components': callable(X,H,Z,t)-> action_vec
      - 'V0': value function at t=0 for each (Z,H,X) index
      - 'policy': policy[t, z, hmask, xmask] -> (rmask, smask)
    
    Assumes actions are binary vectors per patch (0/1).
    Designed for small n (<= ~10).
    """

    n = int(env.patchnum)
    T = int(env.T)
    gamma = 1.0  # your env reward is undiscounted; if you want discounting, change this and DP below.

    # Costs and params from env
    cr = float(env.cr)
    cs = float(env.cs)
    L = float(env.L)
    terminal_penalty = int(env.terminal_penalty)

    # Transition params
    w = np.array(env.w, dtype=float)
    alph0 = float(env.alph0)
    alphZ = float(env.alphZ)
    alphH = float(env.alphH)
    beta0 = float(env.beta0)
    betaH = float(env.betaH)
    qs = float(env.qs)
    deltaH = float(env.deltaH)

    # Z transition (normalize if needed)
    Ztrans = _row_normalize(np.array(env.Ztrans.T, dtype=float))

    # Action sets (bitmask form)
    R_masks = _all_subset_masks_upto(n, kR)
    S_masks = _all_subset_masks_upto(n, kS)

    if not include_do_nothing:
        R_masks = [m for m in R_masks if m != 0]
        S_masks = [m for m in S_masks if m != 0]

    actions: List[Tuple[int, int]] = [(rm, sm) for rm in R_masks for sm in S_masks]

    # Precompute X/H vectors for speed
    Xvec = [_bits_to_vec(xm, n) for xm in range(1 << n)]
    Hvec = [_bits_to_vec(hm, n) for hm in range(1 << n)]

    # Storage
    # V[t, z, hmask, xmask] = optimal expected return from time t onward
    V = np.zeros((T + 1, 2, 1 << n, 1 << n), dtype=np.float64)
    policy = np.zeros((T, 2, 1 << n, 1 << n, 2), dtype=np.int32)  # store (rmask, smask)

    # Absorbing extinction: env terminates immediately when all patches extinct
    # so value from any state with xmask==0 is 0 for all t (already initialized).

    def _reward(xmask_next: int, rmask: int, smask: int, is_terminal: bool) -> float:
        # matches your env.step() reward exactly:
        # reward = (1/n)*sum(X_next) - cr*sum(aR) - cs*sum(aS)
        # terminal: reward -= L * sum(1 - X_next) if terminal_penalty==1
        occ = _popcount(xmask_next) / n
        r = occ - cr * _popcount(rmask) - cs * _popcount(smask)
        if is_terminal and terminal_penalty == 1:
            r -= L * (n - _popcount(xmask_next))
        return r

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # Precompute H transition probabilities given current H and rmask
    # For each hmask and rmask, pH1[i] is P(H_i'==1)
    pH1_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _pH1(hmask: int, rmask: int) -> np.ndarray:
        key = (hmask, rmask)
        if key in pH1_cache:
            return pH1_cache[key]
        H = Hvec[hmask]
        p = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if (rmask >> i) & 1:
                p[i] = 1.0
            else:
                if H[i] == 1:
                    # degrade with prob deltaH; stay high with 1-deltaH
                    p[i] = 1.0 - deltaH
                else:
                    p[i] = 0.0
        pH1_cache[key] = p
        return p

    # Main DP: backward induction
    for t in range(T - 1, -1, -1):
        print(f"Computing V for t={t}...")
        is_terminal = (t == T - 1)

        for z in (0, 1):
            for hmask in range(1 << n):
                H = Hvec[hmask]

                for xmask in range(1 << n):
                    if xmask == 0:
                        # extinct absorbing
                        continue

                    X = Xvec[xmask]

                    # Connectivity and probs depend only on current (X,H,z)
                    connectivity = w @ X  # matches your env: self.w @ X
                    cprob = 1.0 - np.exp(-alph0 * connectivity * (1.0 + alphZ * z) * (1.0 + alphH * H))
                    eprob = _sigmoid(beta0 - betaH * H) * (1.0 - cprob)

                    best_val = -1e18
                    best_action = (0, 0)

                    for rmask, smask in actions:
                        # Occupancy transition probabilities per patch: pX1[i] = P(X_i'==1)
                        pX1 = np.empty(n, dtype=np.float64)
                        for i in range(n):
                            s_i = 1 if ((smask >> i) & 1) else 0
                            if X[i] == 0:
                                # X_next = Bern(cprob) + s_i * Bern(qs)
                                # So P(X_next==1) = 1 - P(both 0)
                                pX1[i] = 1.0 - (1.0 - cprob[i]) * (1.0 - (qs if s_i else 0.0))
                            else:
                                # X_next = Bern(1-eprob) + s_i * Bern(qs)
                                p_surv = 1.0 - eprob[i]
                                pX1[i] = 1.0 - (1.0 - p_surv) * (1.0 - (qs if s_i else 0.0))

                        # Habitat transition probabilities
                        pH1 = _pH1(hmask, rmask)

                        # Expected future value:
                        # Sum over z' (2), x' (2^n), h' (2^n) with product Bernoulli probs.
                        # This is exact and feasible only for small n.
                        exp_future = 0.0

                        # enumerate z'
                        for z2 in (0, 1):
                            pz = Ztrans[z, z2]
                            if pz < prob_prune:
                                continue

                            # enumerate x'
                            for x2 in range(1 << n):
                                px = 1.0
                                # product over i
                                for i in range(n):
                                    xi = (x2 >> i) & 1
                                    pi = pX1[i]
                                    px *= (pi if xi else (1.0 - pi))
                                    if px < prob_prune:
                                        break
                                if px < prob_prune:
                                    continue

                                # episode terminates early in env if x2==0 (extinction)
                                # so continuation value is 0 for remaining steps (matches env)
                                if x2 == 0:
                                    # only immediate reward matters (computed below), future=0
                                    V_cont_x2 = 0.0
                                else:
                                    V_cont_x2 = None  # will sum over h2

                                if x2 == 0:
                                    # we still need expected immediate reward (depends on x2)
                                    # but future is 0 regardless of h2, so we can marginalize h2 out:
                                    # sum_h2 ph(h2) * V[t+1,...] = 0, and reward depends only on x2.
                                    exp_future += pz * px * 0.0
                                    continue

                                # enumerate h'
                                for h2 in range(1 << n):
                                    ph = 1.0
                                    for i in range(n):
                                        hi = (h2 >> i) & 1
                                        pi = pH1[i]
                                        ph *= (pi if hi else (1.0 - pi))
                                        if ph < prob_prune:
                                            break
                                    if ph < prob_prune:
                                        continue

                                    exp_future += pz * px * ph * V[t + 1, z2, h2, x2]

                        # Expected immediate reward depends on x' distribution, not on h'
                        # We'll compute E[r | s,a] by enumerating x' only (cheaper).
                        exp_r = 0.0
                        for x2 in range(1 << n):
                            px = 1.0
                            for i in range(n):
                                xi = (x2 >> i) & 1
                                pi = pX1[i]
                                px *= (pi if xi else (1.0 - pi))
                                if px < prob_prune:
                                    break
                            if px < prob_prune:
                                continue
                            exp_r += px * _reward(x2, rmask, smask, is_terminal)

                        q = exp_r + gamma * exp_future
                        if q > best_val:
                            best_val = q
                            best_action = (rmask, smask)

                    V[t, z, hmask, xmask] = best_val
                    policy[t, z, hmask, xmask, 0] = best_action[0]
                    policy[t, z, hmask, xmask, 1] = best_action[1]
    return {
        #"act": _act,
        #"act_from_components": _act_from_components,
        "V": V,
        "V0": V[0].copy(),
        "policy": policy,
        "Ztrans_used": Ztrans,
        "actions": actions,
        'action_portfolio': env.action_portfolio,
        'envinfo': env.settings,
        "kR": kR,
        "kS": kS,
    }

def _act_from_components(env, policy, T, X: np.ndarray, H: np.ndarray, Z: int, t: int) -> np.ndarray:
    """Return env.action vector with aR and aS filled (binary), aY zeros."""
    # encode
    n = env.patchnum
    xmask = sum((int(X[i]) & 1) << i for i in range(n))
    hmask = sum((int(H[i]) & 1) << i for i in range(n))
    z = int(Z) & 1
    tt = int(t)
    tt = min(max(tt, 0), T - 1)

    rmask = int(policy[tt, z, hmask, xmask, 0])
    smask = int(policy[tt, z, hmask, xmask, 1])

    a = np.zeros(env.actionspace_dim, dtype=int)
    # fill aR, aS (aY zeros)
    aR = np.array([(rmask >> i) & 1 for i in range(n)], dtype=int)
    aS = np.array([(smask >> i) & 1 for i in range(n)], dtype=int)

    if env.action_portfolio == 0:
        a[env.aidx['aR']] = aR
        a[env.aidx['aS']] = aS
    elif env.action_portfolio == 1:
        a[env.aidx['aR']] = aR
    elif env.action_portfolio == 2:
        a[env.aidx['aS']] = aS

    # survey action left as zeros (fully observable controller)
    return a

def _act(env, policy, state_vec: np.ndarray) -> np.ndarray:
    """Takes env.state (or state-like) and returns action vector."""
    X = state_vec[env.sidx['X']]
    H = state_vec[env.sidx['H']]
    Z = state_vec[env.sidx['Z']][0] if 'Z' in env.sidx else 0
    t = int(state_vec[env.sidx['t']][0]) if 't' in env.sidx else 0
    return _act_from_components(env, policy, env.T, X, H, Z, t)

