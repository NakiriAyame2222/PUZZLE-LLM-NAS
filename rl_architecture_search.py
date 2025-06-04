import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy
from tqdm import tqdm

import config as C # import our config
from utils import get_device # 

C.DEVICE = get_device() 

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_attn_choices, num_ffn_choices, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.num_attn_choices = num_attn_choices
        self.num_ffn_choices = num_ffn_choices
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_attn = nn.Linear(hidden_dim, num_attn_choices)
        self.fc_ffn = nn.Linear(hidden_dim, num_ffn_choices)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        attn_logits = self.fc_attn(x)
        ffn_logits = self.fc_ffn(x)
        return attn_logits, ffn_logits

class REINFORCEAgent:
    def __init__(self, state_dim, num_attn_choices, num_ffn_choices, hidden_dim, lr, gamma):
        self.policy_network = PolicyNetwork(state_dim, num_attn_choices, num_ffn_choices, hidden_dim).to(C.DEVICE)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr) # type: ignore
        self.gamma = gamma
        self.num_attn_choices = num_attn_choices
        self.num_ffn_choices = num_ffn_choices
        
        self.saved_log_probs_attn = []
        self.saved_log_probs_ffn = []
        self.rewards = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(C.DEVICE)
        attn_logits, ffn_logits = self.policy_network(state_tensor)
        
        attn_probs = F.softmax(attn_logits, dim=-1)
        ffn_probs = F.softmax(ffn_logits, dim=-1)
        
        attn_dist = Categorical(attn_probs)
        ffn_dist = Categorical(ffn_probs)
        
        action_attn_idx = attn_dist.sample()
        action_ffn_idx = ffn_dist.sample()
        
        self.saved_log_probs_attn.append(attn_dist.log_prob(action_attn_idx))
        self.saved_log_probs_ffn.append(ffn_dist.log_prob(action_ffn_idx))
        
        return action_attn_idx.item(), action_ffn_idx.item()

    def update_policy(self):
        R = 0
        policy_loss_attn = []
        policy_loss_ffn = []
        returns = []

        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(C.DEVICE)
        # Normalize returns for stability (optional but often helpful)
        # returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

        for log_prob_a, log_prob_f, R_t in zip(self.saved_log_probs_attn, self.saved_log_probs_ffn, returns_tensor):
            policy_loss_attn.append(-log_prob_a * R_t)
            policy_loss_ffn.append(-log_prob_f * R_t)
            
        self.optimizer.zero_grad()
        total_policy_loss = torch.cat(policy_loss_attn).sum() + torch.cat(policy_loss_ffn).sum()
        total_policy_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        del self.rewards[:]
        del self.saved_log_probs_attn[:]
        del self.saved_log_probs_ffn[:]


class LLM_NAS_Env:
    def __init__(self, block_library, num_layers, parent_model_config):
        self.block_library = block_library # block_library[layer_idx] = [{'name', 'type', 'trained_obj', 'score', 'cost_latency', ...}, ...]
        self.num_layers = num_layers
        self.parent_model_config = parent_model_config

        # Pre-calculate choices for each layer to define action space
        self.attn_choices_per_layer = []
        self.ffn_choices_per_layer = []
        for i in range(num_layers):
            self.attn_choices_per_layer.append([b for b in block_library[i] if b['type'] == 'attention'])
            self.ffn_choices_per_layer.append([b for b in block_library[i] if b['type'] == 'ffn'])
        
        # State: [current_layer_idx, remaining_latency_budget_norm, remaining_param_mem_budget_norm, remaining_kv_mem_budget_norm]
        # Budgets are normalized 0-1 for stability
        self.state_dim = 1 + 3 # layer_idx + 3 budgets

        self.current_layer_idx = 0
        self.current_architecture = [] # List of {'attention_variant': ..., 'ffn_variant': ...}

        # Budgets
        self.initial_latency_budget = C.TARGET_LATENCY_MAX
        self.initial_param_mem_budget = C.TARGET_MEMORY_MAX_GB 
        # KV cache is tricky. Base it on parent and then modify by factors.
        self.initial_kv_mem_budget_approx = C.TARGET_MEMORY_MAX_GB * 0.5 # Rough estimate of KV portion of budget
        # A more accurate way:
        # base_kv_per_layer_parent = C.KV_CACHE_MEMORY_PER_TOKEN_PER_HEAD_PER_LAYER * \
        #                     parent_model_config.num_attention_heads * C.MAX_SEQ_LENGTH * C.GKD_BATCH_SIZE
        # self.initial_kv_mem_budget = base_kv_per_layer_parent * num_layers * C.TARGET_KV_BUDGET_FACTOR # Factor of parent


        self.remaining_latency = 0
        self.remaining_param_mem = 0
        self.remaining_kv_mem = 0
        
    def _get_max_choices_for_policy(self):
        """Helper to define fixed size policy network based on max choices across layers"""
        max_attn = 0
        max_ffn = 0
        for i in range(self.num_layers):
            max_attn = max(max_attn, len(self.attn_choices_per_layer[i]))
            max_ffn = max(max_ffn, len(self.ffn_choices_per_layer[i]))
        return max_attn, max_ffn

    def reset(self):
        self.current_layer_idx = 0
        self.current_architecture = []
        self.remaining_latency = self.initial_latency_budget
        self.remaining_param_mem = self.initial_param_mem_budget
        self.remaining_kv_mem = self.initial_kv_mem_budget_approx 
        return self._get_state()

    def _get_state(self):
        # Normalize budgets
        lat_norm = self.remaining_latency / self.initial_latency_budget if self.initial_latency_budget > 0 else 0
        param_norm = self.remaining_param_mem / self.initial_param_mem_budget if self.initial_param_mem_budget > 0 else 0
        kv_norm = self.remaining_kv_mem / self.initial_kv_mem_budget_approx if self.initial_kv_mem_budget_approx > 0 else 0
        
        return np.array([
            self.current_layer_idx / self.num_layers, # Normalize layer index
            max(0, lat_norm), # Ensure non-negative
            max(0, param_norm),
            max(0, kv_norm)
        ], dtype=np.float32)

    def step(self, action_attn_idx, action_ffn_idx):
        # Action indices are for the *current layer's* specific list of choices
        # The policy network might be designed for max_choices, so clipping might be needed if some layers have fewer.
        # However, if PolicyNetwork is recreated or adapted per layer, this is fine.
        # For this example, assume PolicyNetwork handles variable choices or agent clips action based on current layer.
        
        current_attn_options = self.attn_choices_per_layer[self.current_layer_idx]
        current_ffn_options = self.ffn_choices_per_layer[self.current_layer_idx]

        # Ensure action indices are valid for current layer
        actual_attn_idx = min(action_attn_idx, len(current_attn_options) - 1)
        actual_ffn_idx = min(action_ffn_idx, len(current_ffn_options) - 1)

        chosen_attn_variant = current_attn_options[actual_attn_idx]
        chosen_ffn_variant = current_ffn_options[actual_ffn_idx]

        self.current_architecture.append({
            'layer_idx': self.current_layer_idx,
            'attention_variant': chosen_attn_variant,
            'ffn_variant': chosen_ffn_variant
        })

        # Update budgets
        self.remaining_latency -= (chosen_attn_variant['cost_latency'] + chosen_ffn_variant['cost_latency'])
        self.remaining_param_mem -= (chosen_attn_variant['cost_param_memory'] + chosen_ffn_variant['cost_param_memory'])
        
        # Approximate KV cache update
        base_kv_per_layer_parent = C.KV_CACHE_MEMORY_PER_TOKEN_PER_HEAD_PER_LAYER * \
                            self.parent_model_config.num_attention_heads * C.MAX_SEQ_LENGTH * C.GKD_BATCH_SIZE
        self.remaining_kv_mem -= (chosen_attn_variant['cost_kv_cache_factor'] * base_kv_per_layer_parent)


        self.current_layer_idx += 1
        done = self.current_layer_idx >= self.num_layers
        
        reward = 0
        if done:
            reward = self._calculate_final_reward()
        # else:
            # Optional: small intermediate reward/penalty if desired
            # reward = -0.01 # Small penalty for each step to encourage shorter valid paths (if applicable)

        return self._get_state(), reward, done

    def _calculate_final_reward(self):
        total_quality_score = 0
        total_latency_used = 0
        total_param_mem_used = 0
        total_kv_mem_factor_sum = 0 # Sum of factors

        for layer_choice in self.current_architecture:
            # Remember: lower KL is better, so score might be -KL or 1/KL for maximization
            # Assuming 'score' from block_library is already "higher is better"
            # If 'score' is KL_div (lower is better), then:
            # total_quality_score -= layer_choice['attention_variant']['score'] 
            # total_quality_score -= layer_choice['ffn_variant']['score']
            # For this example, let's assume score in block_library is KL (lower is better)
            # So we want to minimize it. Reward will be -sum(scores)
            total_quality_score -= (layer_choice['attention_variant']['score'] + layer_choice['ffn_variant']['score'])


            total_latency_used += (layer_choice['attention_variant']['cost_latency'] + layer_choice['ffn_variant']['cost_latency'])
            total_param_mem_used += (layer_choice['attention_variant']['cost_param_memory'] + layer_choice['ffn_variant']['cost_param_memory'])
            total_kv_mem_factor_sum += layer_choice['attention_variant']['cost_kv_cache_factor']

        base_kv_per_layer_parent = C.KV_CACHE_MEMORY_PER_TOKEN_PER_HEAD_PER_LAYER * \
                                   self.parent_model_config.num_attention_heads * C.MAX_SEQ_LENGTH * C.GKD_BATCH_SIZE
        total_kv_mem_used_approx = total_kv_mem_factor_sum * base_kv_per_layer_parent


        reward = total_quality_score # This is negative sum of KLs, so higher (less negative) is better.

        # Penalties for exceeding budget
        budget_exceeded = False
        if self.remaining_latency < 0: # total_latency_used > self.initial_latency_budget:
            reward -= C.RL_BUDGET_PENALTY * abs(self.remaining_latency / self.initial_latency_budget)
            budget_exceeded = True
        if self.remaining_param_mem < 0: # total_param_mem_used > self.initial_param_mem_budget:
            reward -= C.RL_BUDGET_PENALTY * abs(self.remaining_param_mem / self.initial_param_mem_budget)
            budget_exceeded = True
        if self.remaining_kv_mem < 0: # total_kv_mem_used_approx > self.initial_kv_mem_budget_approx:
             reward -= C.RL_BUDGET_PENALTY * abs(self.remaining_kv_mem / self.initial_kv_mem_budget_approx)
             budget_exceeded = True
        
        # Penalty for resource usage even if within budget (to encourage efficiency)
        # Normalize usage by initial budget
        if not budget_exceeded: # Only apply if not already heavily penalized
            norm_latency_usage = total_latency_used / self.initial_latency_budget if self.initial_latency_budget > 0 else 0
            norm_param_usage = total_param_mem_used / self.initial_param_mem_budget if self.initial_param_mem_budget > 0 else 0
            norm_kv_usage = total_kv_mem_used_approx / self.initial_kv_mem_budget_approx if self.initial_kv_mem_budget_approx > 0 else 0
            
            reward -= C.RL_RESOURCE_USAGE_PENALTY_FACTOR * (norm_latency_usage + norm_param_usage + norm_kv_usage)
            
        return reward


def run_rl_search(block_library, parent_model_config):
    num_layers = len(block_library)
    env = LLM_NAS_Env(block_library, num_layers, parent_model_config)
    
    # The PolicyNetwork needs to know the max number of choices for attn/ffn across all layers
    # This is a simplification. A more robust agent (e.g. RNN controller) could handle variable choices better.
    max_attn_choices, max_ffn_choices = env._get_max_choices_for_policy()
    if max_attn_choices == 0 or max_ffn_choices == 0:
        print("Error: No attention or FFN choices found in block library for RL.")
        return None

    agent = REINFORCEAgent(state_dim=env.state_dim, 
                           num_attn_choices=max_attn_choices, 
                           num_ffn_choices=max_ffn_choices,
                           hidden_dim=C.RL_AGENT_HIDDEN_DIM, 
                           lr=C.RL_LEARNING_RATE, 
                           gamma=C.RL_GAMMA)

    best_reward = -float('inf')
    best_architecture = None

    for i_episode in range(C.RL_NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for t in range(num_layers): # Max steps = num_layers
            # Here, we need to ensure the agent's action output (0 to max_choices-1)
            # is valid for the *current* layer's actual number of choices.
            # The LLM_NAS_Env.step method already handles clipping/min.
            action_attn_idx, action_ffn_idx = agent.select_action(state)
            
            next_state, reward, done = env.step(action_attn_idx, action_ffn_idx)
            agent.rewards.append(reward) # Store reward for this step (will be 0 until done)
            episode_reward += reward # This will be the final reward
            state = next_state
            if done:
                break
        
        agent.update_policy() # Update at the end of the episode

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_architecture = copy.deepcopy(env.current_architecture)
            print(f"Episode {i_episode+1}: New best reward: {best_reward:.4f}")
        
        if (i_episode + 1) % 10 == 0:
            print(f"Episode {i_episode+1}: Total Reward: {episode_reward:.4f}")
            if best_architecture:
                 # Print a snippet of the best architecture so far
                print(f"  Best arch (L0): A='{best_architecture[0]['attention_variant']['name']}', F='{best_architecture[0]['ffn_variant']['name']}'")


    print(f"\nRL Search Finished. Best Reward: {best_reward:.4f}")
    return best_architecture