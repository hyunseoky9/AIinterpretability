import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, CosineAnnealingLR
from torch.distributions import Beta, Dirichlet, Categorical, Independent, Bernoulli

class ActorCritic_metapop1_MDP(nn.Module):
    def __init__(self, input_dims,
                actor_hidden_size, actor_hidden_num, 
                critic_hidden_size, critic_hidden_num,
                encoder_hidden_size, encoder_hidden_num, encoder_output_dims,
                actor_lr, critic_lr, encoder_lr,
                min_lr, lrdecaytype, lrdecayrate,
                scheduler_info, device, entropy_loss, info):
        '''
        Actor-Critic architecture for metapop1 without partial observability (full observation of occupancy and dispersal regime)
        '''
        super(ActorCritic_metapop1_MDP, self).__init__()
        self.entropy_loss = entropy_loss

        # environment-specific info        
        self.envinfo = info
        self.npatches = info['npatches']
        self.kR = info['kR']
        self.kS = info['kS']
        self.Rheadsize = info['Rheadsize']
        self.Sheadsize = info['Sheadsize']
        self.Rbernoulli = info['Rbernoulli']
        self.Sbernoulli = info['Sbernoulli']

        # build the encoder model
        feature_dims = input_dims[-1]
        layers = [nn.Linear(feature_dims, encoder_hidden_size[0]), nn.ReLU()]
        for i in range(encoder_hidden_num - 1):
            layers.append(nn.Linear(encoder_hidden_size[i], encoder_hidden_size[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(encoder_hidden_size[-1], encoder_output_dims))
        self.encoder = nn.Sequential(*layers)

        # build the actor model
        layers = [nn.Linear(encoder_output_dims*2, actor_hidden_size[0]), nn.ReLU()] # # *2 because we concatenate the global pooling output as well. 
        for i in range(actor_hidden_num - 1):
            layers.append(nn.Linear(actor_hidden_size[i], actor_hidden_size[i + 1]))
            layers.append(nn.ReLU())
        self.actor = nn.Sequential(*layers)

        self.R_head = nn.Linear(actor_hidden_size[-1], 1)
        self.S_head = nn.Linear(actor_hidden_size[-1], 1)
        if self.Rbernoulli == 0:
            self.R_stop = nn.Linear(encoder_output_dims, 1)
        if self.Sbernoulli == 0:
            self.S_stop = nn.Linear(encoder_output_dims, 1)

        # build the critic model
        layers = [nn.Linear(encoder_output_dims, critic_hidden_size[0]), nn.ReLU()]
        for i in range(critic_hidden_num - 1):
            layers.append(nn.Linear(critic_hidden_size[i], critic_hidden_size[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(critic_hidden_size[-1], 1))
        self.critic = nn.Sequential(*layers)

        # set up optimizer and device
        param_groups = [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.actor.parameters(), "lr": actor_lr},
            {"params": self.critic.parameters(), "lr": critic_lr},
            {"params": self.R_head.parameters(), "lr": actor_lr},
            {"params": self.S_head.parameters(), "lr": actor_lr},
        ]
        if self.Rbernoulli == 0:
            param_groups.append({"params": self.R_stop.parameters(), "lr": actor_lr})
        if self.Sbernoulli == 0:
            param_groups.append({"params": self.S_stop.parameters(), "lr": actor_lr})

        self.optimizer = optim.Adam(param_groups, eps=1e-8)

        self.device = T.device(device)
        self.to(self.device)
        # set up learning rate scheduler
        if lrdecaytype == 'exp':
            self.scheduler = ExponentialLR(self.optimizer, gamma=lrdecayrate)
        elif lrdecaytype == 'multistep':
            self.scheduler = MultiStepLR(self.optimizer, milestones=scheduler_info['lr_drop_ep'],
                                          gamma=scheduler_info['lr_drop_gamma'])
        else:
            raise ValueError(f"Unknown lrdecaytype: {lrdecaytype}")
    
    def encode(self, state):
        # state is [B, N, feature_dim]
        x = self.encoder(state) # [B, N, encoder_output_dims]
        g = T.mean(x, dim=1) # global pooling [B, encoder_output_dims]
        return x, g
    
    def actor_forward(self, encoded_state, global_pool):
        g_rep = global_pool.unsqueeze(1).expand(-1, encoded_state.size(1), -1)
        actor_in = T.cat([encoded_state, g_rep], dim=2)
        x0 = self.actor(actor_in) # [B, N, actor_hidden_size[-1]]
        Rlogits = self.R_head(x0).squeeze(-1)
        Slogits = self.S_head(x0).squeeze(-1)
        if self.Rbernoulli == 0:
            Rstop = self.R_stop(global_pool)
            Rlogits = T.cat([Rlogits, Rstop], dim=1)
        if self.Sbernoulli == 0:
            Sstop = self.S_stop(global_pool)
            Slogits = T.cat([Slogits, Sstop], dim=1)
        logits = T.cat([Rlogits, Slogits], dim=1) # [B, Rheadsize+Sheadsize]
        return logits

    def critic_forward(self, global_pool):
        x = self.critic(global_pool)
        return x
    
    def forward(self, state):
        # get both actor and critic logits
        encoded_state, global_pool = self.encode(state)

        g_rep = global_pool.unsqueeze(1).expand(-1, encoded_state.size(1), -1)
        actor_in = T.cat([encoded_state, g_rep], dim=2)
        x0 = self.actor(actor_in) # [B, N, actor_hidden_size[-1]]
        Rlogits = self.R_head(x0).squeeze(-1)
        Slogits = self.S_head(x0).squeeze(-1)
        if self.Rbernoulli == 0:
            Rstop = self.R_stop(global_pool)
            Rlogits = T.cat([Rlogits, Rstop], dim=1)
        if self.Sbernoulli == 0:
            Sstop = self.S_stop(global_pool)
            Slogits = T.cat([Slogits, Sstop], dim=1)
        logits = T.cat([Rlogits, Slogits], dim=1) # [B, Rheadsize+Sheadsize]
        value = self.critic(global_pool)
        return logits, value

    def getdist(self, x):
        # process beta dist paramgeters
        if self.Rbernoulli == 1:
            xR = T.sigmoid(x[:, :self.Rheadsize])
            Rdist = Independent(Bernoulli(probs=xR), 1)
        else:
            Rdist = None
        if self.Sbernoulli == 1:
            xS = T.sigmoid(x[:, self.Rheadsize:self.Rheadsize+self.Sheadsize])
            Sdist = Independent(Bernoulli(probs=xS), 1)
        else:
            Sdist = None
        return Rdist, Sdist
    
    def sample_wo_replacement(self, logits, k):
        """
        logits: [1, n+1] including STOP
        k: max number of selections
        """
        stop_idx = logits.shape[1] - 1
        # Gumbel noise
        g = -T.log(-T.log(T.rand_like(logits)))
        scores = logits + g
        ranks = T.argsort(scores, dim=1, descending=True)
        stop_idx_chosen = False
        chosen = []
        for idx in ranks[0]:
            idx = idx.item()
            if idx == stop_idx:
                stop_idx_chosen = True
                break
            chosen.append(idx)
            if len(chosen) >= k:
                break

        a = T.zeros(self.npatches, device=logits.device)
        if len(chosen) > 0:
            a[T.tensor(chosen, device=logits.device, dtype=T.long)] = 1
        a = a.unsqueeze(0)
        chosen_idx = chosen + ([stop_idx] if stop_idx_chosen else [])
        chosen_idx_padded = chosen_idx + [-1] * (k - len(chosen_idx))
        chosen_len = len(chosen_idx)
        return a, T.tensor(chosen_idx_padded, device=logits.device, dtype=T.long), chosen_len

    def logprob_entropy_wo_replacement_batched(self, logits, seq_idx, seq_len):

        """
        logits:  (B, N) logits over patches + STOP
        seq_idx: (B, Lmax) long, padded with -1
        seq_len: (B,) number of valid steps (includes STOP if it was sampled)
        Returns:
        logp: (B,)
        ent:  (B,) (0 if entropy_loss=False)
        """
        B, N = logits.shape
        Lmax = seq_idx.shape[1]
        device = logits.device

        # remaining mask: True means available
        remaining = T.ones((B, N), dtype=T.bool, device=device)

        logp = T.zeros((B,), device=device)
        ent  = T.zeros((B,), device=device)

        for t in range(Lmax):
            # which batch elements are still active at this step?
            active = t < seq_len
            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            if active_idx.numel() == 0:
                break

            # slice only active rows.
            rem_a = remaining[active_idx]
            logits_a = logits[active_idx]
            # mask invalid choices
            step_logits_a = logits_a.masked_fill(~rem_a, float("-inf"))
            # compute log-probs and probs
            log_probs_a = F.log_softmax(step_logits_a, dim=-1)

            if self.entropy_loss:
                # entropy of categorical over remaining (per batch)
                # H = -sum p log p, but ignore -inf entries (p=0)
                probs_a = log_probs_a.exp()
                masked_probs_a = probs_a.masked_fill(~rem_a, 0.0)
                masked_log_probs_a = log_probs_a.masked_fill(~rem_a, 0.0)
                ent_step_a = -(masked_probs_a * masked_log_probs_a).sum(dim=-1)
                ent[active_idx] += ent_step_a

            # gather chosen indices for active rows
            idx_t_a = seq_idx[active_idx, t]
            valid_a = idx_t_a >= 0
            if valid_a.any():
                chosen_a = idx_t_a[valid_a]
                logp[active_idx[valid_a]] += log_probs_a[valid_a, chosen_a]
                # remove chosen items (no replacement)
                remaining[active_idx[valid_a], chosen_a] = False
        return logp, ent

    def logprob_entropy_without_replacement(self, logits, sampled_idx, seqlen, eps=1e-12, entropycalc=False):
        p = F.softmax(logits, dim=1).squeeze(0).clone()
        logp = T.tensor(0.0, device=logits.device)
        entropy = T.tensor(0.0, device=logits.device)
        sampled_idx = sampled_idx[:seqlen]
        for idx in sampled_idx:
            z = p.sum()
            p_norm = (p / z).clamp_min(eps)
            if self.entropy_loss and entropycalc:
                entropy = entropy + (-(p_norm * T.log(p_norm)).sum())
            
            prob = p_norm[idx].clamp_min(eps)
            logp = logp + T.log(prob)
            p[idx] = 0.0
        return logp, entropy

    def getaction(self, state, get_action_only=False, withvalue=False):
        if withvalue:
            x, value = self(state)
        else:
            encoded_state, global_pool = self.encode(state)
            x = self.actor_forward(encoded_state, global_pool)
        Rdist, Sdist = self.getdist(x)
        if self.Rbernoulli == 1:
            aR = Rdist.sample()
            sampledR = T.tensor([], device=state.device, dtype=T.long)
            seqlenR = 0
        else:
            aR, sampledR, seqlenR = self.sample_wo_replacement(x[:, 0:self.Rheadsize], self.kR)
        if self.Sbernoulli == 1:
            aS = Sdist.sample()
            sampledS = T.tensor([], device=state.device, dtype=T.long)
            seqlenS = 0
        else:
            aS, sampledS, seqlenS = self.sample_wo_replacement(x[:, self.Rheadsize:self.Rheadsize+self.Sheadsize], self.kS)
        action = T.cat([aR, aS], dim=1)

        if get_action_only:
            return action
        
        if self.Rbernoulli == 1:
            logprobR = Rdist.log_prob(aR).item()
        else:
            logprobR, _ = self.logprob_entropy_without_replacement(x[:, 0:self.Rheadsize], sampledR, seqlenR, entropycalc=False)
        if self.Sbernoulli == 1:
            logprobS = Sdist.log_prob(aS).item()
        else: 
            logprobS, _ = self.logprob_entropy_without_replacement(x[:, self.Rheadsize:self.Rheadsize+self.Sheadsize], sampledS, seqlenS, entropycalc=False)
        logprob = logprobR + logprobS
        sampleinfo = T.cat([
            sampledR,
            sampledS,
            T.tensor([seqlenR, seqlenS], device=state.device, dtype=T.long)
        ], dim=0)  
        if withvalue:
            return action, logprob, value, sampleinfo # sampleinfo contains wo-replacement patchid idxs (or stop idx) for R and S (kR + kS), and number of idx's sampled for R and S (2)
        else:
            return action, logprob, sampleinfo
    
    def get_log_prob(self, encoded_state, global_pool, actions, infoNbatch):
        '''
        sampled_info = (info_arr, batch) where info_arr is any info and batch is the indices for the current minibatch.
        '''
        info_arr = infoNbatch[0] # select the info for the current batch
        batch = infoNbatch[1]
        x = self.actor_forward(encoded_state, global_pool)
        Rdist, Sdist = self.getdist(x)
        sampled_info = T.stack(info_arr, dim=0).to(encoded_state.device, dtype=T.long)[batch]
        if self.Rbernoulli == 1:
            logprobR = Rdist.log_prob(actions[:, :self.npatches])
            entropyR = Rdist.entropy()
        else:
            logprobR, entropyR = self.logprob_entropy_wo_replacement_batched(
                x[:, 0:self.Rheadsize],
                sampled_info[:,:self.kR], 
                sampled_info[:,-2])
        if self.Sbernoulli == 1:
            logprobS = Sdist.log_prob(actions[:, self.npatches:self.npatches+self.Sheadsize])
            entropyS = Sdist.entropy()
        else:
            Ridx = 0 if self.Rbernoulli == 1 else self.kR
            logprobS, entropyS = self.logprob_entropy_wo_replacement_batched(
                x[:, self.Rheadsize:self.Rheadsize+self.Sheadsize],
                sampled_info[:,Ridx: Ridx+self.kS],
                sampled_info[:,-1])
        return logprobR + logprobS, entropyR + entropyS

    def deterministic_sample_without_replacement(self, logits, k):
        stop_idx = logits.shape[1] - 1
        sorted_indices = T.argsort(logits, dim=1, descending=True)

        # first position where STOP appears (if any)
        pos = (sorted_indices == stop_idx).nonzero(as_tuple=False)
        stop_pos = pos[0,1].item() if pos.numel() > 0 else logits.shape[1]
        cut_idx = min(k, stop_pos)
        chosen_indices = sorted_indices[0][:cut_idx]
        a = T.zeros(self.npatches, device=logits.device)
        a[chosen_indices] = 1
        return a

    def get_deterministic_action(self, state):
        encoded_state, global_pool = self.encode(state)
        x = self.actor_forward(encoded_state, global_pool)
        
        if self.Rbernoulli == 1:
            aR = (T.sigmoid(x[:, 0:self.Rheadsize]) > 0.5).float()
        else:
            aR = self.deterministic_sample_without_replacement(x[:, 0:self.Rheadsize], self.kR).unsqueeze(0)
        if self.Sbernoulli == 1:
            aS = (T.sigmoid(x[:, self.Rheadsize:self.Rheadsize+self.Sheadsize]) > 0.5).float()
        else:
            aS = self.deterministic_sample_without_replacement(x[:, self.Rheadsize:self.Rheadsize+self.Sheadsize], self.kS).unsqueeze(0)
        action = T.cat([aR, aS], dim=1)
        return action
    