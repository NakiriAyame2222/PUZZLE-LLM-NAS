# Model Configuration
PARENT_MODEL_NAME = "gpt2" # Or a small Llama variant if easily accessible and modifiable
# PARENT_MODEL_NAME = "meta-llama/Llama-2-7b-hf" # Ambitious, needs resources

# Search Space
# For GPT-2, num_heads is fixed per layer, but we can simulate GQA by reducing KV heads
# For simplicity, we'll focus on FFN reduction and no-ops first.
ATTENTION_VARIANTS_CONFIG = {
    "num_kv_heads_options": [None, 4, 2, 1], # None means use original, others for GQA
    "use_linear_attention": True,
    "no_op_attention": True,
}
FFN_VARIANTS_CONFIG = {
    "intermediate_reduction_factors": [None, 0.87, 0.75, 0.5, 0.25, 0.1], # None means original
    "use_linear_ffn": True,
    "no_op_ffn": True,
}

# BLD Configuration
BLD_EPOCHS = 1
BLD_BATCH_SIZE = 8
BLD_LR = 5e-5
BLD_VALIDATION_SET_SIZE = 1000 # For replace-1-block scoring
DISTILLATION_MIX_PATH = "wikitext" # Example dataset, paper uses FineWeb, Dolma, Buzz
DISTILLATION_MIX_SUBSET = "wikitext-2-raw-v1"
MAX_SEQ_LENGTH = 128 # Keep it small for faster dev

# MIP Configuration
TARGET_THROUGHPUT_MIN = 100 # tokens/sec (simulated)
TARGET_LATENCY_MAX = 500    # ms/batch (simulated)
TARGET_MEMORY_MAX_GB = 4    # GB (simulated)

# GKD Configuration
GKD_EPOCHS = 1
GKD_BATCH_SIZE = 8
GKD_LR = 1e-5
GKD_NUM_TOKENS = 1_000_000 # Paper uses 45B, we use much less
KLD_WEIGHT = 1.0
COSINE_SIM_WEIGHT = 1.0

# Hardware (Simulation)
# Simulated cost per original attention/FFN op (e.g., in FLOPs or arbitrary units)
# These would be measured on target hardware in the real paper
BASE_ATTENTION_COST = 10
BASE_FFN_COST = 12
BASE_ATTENTION_MEMORY = 0.1 # GB
BASE_FFN_MEMORY = 0.15    # GB
KV_CACHE_MEMORY_PER_TOKEN_PER_HEAD_PER_LAYER = 0.0001 # GB (very rough)

# RL Configuration
RL_AGENT_HIDDEN_DIM = 64
RL_LEARNING_RATE = 1e-3
RL_NUM_EPISODES = 100 # 实际应用中需要更多
RL_GAMMA = 0.99 # Discount factor for future rewards (if needed, often 1.0 for NAS)
RL_BUDGET_PENALTY = 1000 # Large penalty for exceeding budget
RL_RESOURCE_USAGE_PENALTY_FACTOR = 0.1 # Small penalty for using resources