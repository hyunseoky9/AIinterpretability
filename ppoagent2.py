import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class PPOMemory:
    def __init__(self, minibatch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.info = []

        self.minibatch_size = minibatch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.minibatch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.minibatch_size] for i in batch_start]

        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.dones), \
                self.info, \
                batches
    
    def store_memory(self, state, prob, val, action, reward, done, info=None):
        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.info.append(info)
    
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.info = []

class PPOAgent2:
    def __init__(self, c1, c2, entropy_loss, 
                 minibatch_size,
                 policy_clip,
                 gamma, gae_lambda,
                 n_epochs,
                 adv_normalization,
                 KL_stopping, target_KL,
                 actorcritic):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.c1 = c1
        self.c2 = c2
        self.entropy_loss = entropy_loss
        self.actorcritic = actorcritic
        self.KL_stopping = KL_stopping
        self.target_KL = target_KL
        self.memory = PPOMemory(minibatch_size)
        self.adv_normalization = adv_normalization

    def remember(self, state, action, probs, vals, reward, done, info=None):
        self.memory.store_memory(state, probs, vals, action, reward, done, info)
    
    def save_checkpoint(self, network,path):
        T.save(network, path)

    def save_models(self,path):
        self.save_checkpoint(self.actorcritic, path)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actorcritic.device)
        
        action, logprob, value, info = self.actorcritic.getaction(state, withvalue=True)
    
        probs = T.squeeze(logprob).item()
        action = T.squeeze(action).cpu().detach().numpy()
        value = T.squeeze(value).item()

        return action, probs, value, info

    def compute_gae_1d(self, rewards,values,dones,gamma ,lam ,last_value=0.0):
        """
        O(T) GAE(λ) for a single trajectory or a concatenated rollout.

        Args:
            rewards: shape [T]
            values:  shape [T]  (V(s_t) stored during rollout)
            dones:   shape [T]  (1 if terminal at t else 0)
            gamma: discount factor
            lam: GAE lambda
            last_value: V(s_{T}) for bootstrapping the final step if not terminal.
                        If you don't have V(s_T), pass 0 and this still behaves sensibly.

        Returns:
            advantages: shape [T]
            returns:    shape [T] where returns = advantages + values (GAE-style target)
        """
        T_len = len(rewards)
        advantages = np.zeros(T_len, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T_len)):
            nonterminal = 1.0 - float(dones[t])

            next_value = last_value if t == T_len - 1 else values[t + 1]

            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae

            # If this timestep ended an episode, reset the accumulator
            if dones[t]:
                gae = 0.0

        returns = advantages + values.astype(np.float32)
        return advantages, returns

    def learn(self):
        # keep track of losses for logging
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_sum = 0.0
        n_minibatches = 0

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, done_arr, info_arr, batches = \
                self.memory.generate_batches()
            
            values = vals_arr
            last_value = 0
            advantages, returns = self.compute_gae_1d(reward_arr,values,done_arr,
                                                      self.gamma,self.gae_lambda,last_value)
            # Advantage normalization (once per epoch, before minibatches)
            if self.adv_normalization:
                advantages = (advantages - advantages.mean()) / (advantages.std(ddof=0) + 1e-10)

            kl_running = 0.0
            kl_count = 0
            stop_early = False
            for batch in batches:
                # convert batch data to tensors
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actorcritic.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actorcritic.device)
                actions = T.tensor(action_arr[batch]).to(self.actorcritic.device)
                # encode states with new network
                encoded_state, global_pool = self.actorcritic.encode(states)
                # calculate critic value with new network
                critic_value = self.actorcritic.critic_forward(global_pool)
                critic_value = T.squeeze(critic_value)
                # calculate new log probs with new network
                new_probs, current_entropy = self.actorcritic.get_log_prob(encoded_state, global_pool, actions, (info_arr, batch))
                prob_ratio = (new_probs - old_probs).exp()

                # KL estimate
                approx_kl = (old_probs - new_probs).mean()
                kl_running += approx_kl.item()
                kl_count += 1
                # get advantages and returns for the minibatch and convert to tensors
                advantage = T.tensor(advantages[batch], dtype=T.float, device=self.actorcritic.device)
                returns_t = T.tensor(returns[batch], dtype=T.float, device=self.actorcritic.device)
                # calculate actor loss
                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                # calculate critic loss
                critic_loss = (returns_t - critic_value).pow(2).mean()
                # calculate total loss
                total_loss = actor_loss + self.c1 * critic_loss
                # add entropy loss if using
                if self.entropy_loss:
                    entropy_loss = -self.c2 * current_entropy.mean()
                    total_loss += entropy_loss
                # accumulate losses for logging
                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                entropy_sum += current_entropy.mean().item()
                n_minibatches += 1

                # take a gradient step
                self.actorcritic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actorcritic.parameters(), 1.0)
                self.actorcritic.optimizer.step()


                # KL early stopping check 
                if (self.KL_stopping) and (approx_kl.item() > self.target_KL):
                    stop_early = True
                    break
            if stop_early:
                break
        mean_actor_loss = actor_loss_sum / max(n_minibatches, 1)
        mean_critic_loss = critic_loss_sum / max(n_minibatches, 1)
        mean_entropy = entropy_sum / max(n_minibatches, 1)




        self.memory.clear_memory() # clear memory after learning is done before next round of data collection
        return mean_actor_loss, mean_critic_loss, mean_entropy