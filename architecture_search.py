from mip import Model, xsum, BINARY, OptimizationStatus # type: ignore
import config as C

def solve_mip(block_library, parent_model_config):
    num_layers = len(block_library)
    m = Model("llm_puzzle")

    # Decision variables: x[layer_idx][block_choice_idx]
    # We need to combine attention and FFN choices for each layer.
    # The block_library is flat per layer: [ {attn_opt1}, {ffn_opt1}, {attn_opt2}, ... ]
    # We need to make pairs for MIP: one attention choice + one FFN choice per layer.
    
    # Preprocess block_library: group by attention and FFN for each layer
    layer_options = [] # List of lists of (attn_choice_idx, ffn_choice_idx)
    
    # This representation is simpler: each "item" in MIP is a full layer config
    # x[i][j] = 1 if layer i uses combined option j
    # Option j is a specific (attention_variant, ffn_variant) pair.

    # Let's simplify: MIP chooses one attention block AND one FFN block per layer
    x_attn = [[m.add_var(var_type=BINARY, name=f"x_attn_L{i}_V{j}") 
               for j, var_info in enumerate(block_library[i]) if var_info['type'] == 'attention'] 
              for i in range(num_layers)]
    
    x_ffn = [[m.add_var(var_type=BINARY, name=f"x_ffn_L{i}_V{j}") 
              for j, var_info in enumerate(block_library[i]) if var_info['type'] == 'ffn']
             for i in range(num_layers)]

    # Store actual variant info corresponding to x_attn/x_ffn indices
    attn_variants_for_mip = [[var_info for var_info in block_library[i] if var_info['type'] == 'attention'] for i in range(num_layers)]
    ffn_variants_for_mip = [[var_info for var_info in block_library[i] if var_info['type'] == 'ffn'] for i in range(num_layers)]


    # Objective: Minimize sum of KL divergence scores
    # (Since lower KL is better)
    objective_terms = []
    for i in range(num_layers):
        for j, var_info in enumerate(attn_variants_for_mip[i]):
            objective_terms.append(var_info['score'] * x_attn[i][j])
        for j, var_info in enumerate(ffn_variants_for_mip[i]):
            objective_terms.append(var_info['score'] * x_ffn[i][j])
    m.objective = xsum(objective_terms)


    # Constraints:
    # 1. Select exactly one attention variant per layer
    for i in range(num_layers):
        m += xsum(x_attn[i][j] for j in range(len(x_attn[i]))) == 1, f"one_attn_L{i}"

    # 2. Select exactly one FFN variant per layer
    for i in range(num_layers):
        m += xsum(x_ffn[i][j] for j in range(len(x_ffn[i]))) == 1, f"one_ffn_L{i}"
        
    # 3. Total Latency Constraint (sum of chosen block latencies)
    total_latency_expr = []
    for i in range(num_layers):
        for j, var_info in enumerate(attn_variants_for_mip[i]):
            total_latency_expr.append(var_info['cost_latency'] * x_attn[i][j])
        for j, var_info in enumerate(ffn_variants_for_mip[i]):
            total_latency_expr.append(var_info['cost_latency'] * x_ffn[i][j])
    m += xsum(total_latency_expr) <= C.TARGET_LATENCY_MAX, "max_latency"

    # 4. Total Parameter Memory Constraint
    total_param_mem_expr = []
    for i in range(num_layers):
        for j, var_info in enumerate(attn_variants_for_mip[i]):
            total_param_mem_expr.append(var_info['cost_param_memory'] * x_attn[i][j])
        for j, var_info in enumerate(ffn_variants_for_mip[i]):
            total_param_mem_expr.append(var_info['cost_param_memory'] * x_ffn[i][j])
    # KV cache memory depends on attention choices (num_kv_heads)
    # This is a simplified sum for now. Needs num_heads from chosen attn blocks.
    # We'll add a placeholder for KV cache.
    # total_kv_cache_mem = calculate_kv_cache_memory(...) # This needs chosen attn blocks!
    # For now, let's use a factor from the 'cost_kv_cache_factor'
    
    # Placeholder for KV cache calculation within MIP (complex) or approximate it
    # Simplified: sum param memory + sum (kv_cache_factor * base_kv_mem_per_layer)
    total_kv_mem_expr = []
    base_kv_per_layer = C.KV_CACHE_MEMORY_PER_TOKEN_PER_HEAD_PER_LAYER * \
                        parent_model_config.num_attention_heads * C.MAX_SEQ_LENGTH * C.GKD_BATCH_SIZE # Approx
    
    for i in range(num_layers):
        for j, var_info in enumerate(attn_variants_for_mip[i]):
            total_kv_mem_expr.append(var_info['cost_kv_cache_factor'] * base_kv_per_layer * x_attn[i][j])
            
    m += xsum(total_param_mem_expr) + xsum(total_kv_mem_expr) <= C.TARGET_MEMORY_MAX_GB, "max_memory"


    # 5. Throughput Constraint: (batch_size * seq_len) / total_runtime >= min_throughput
    # total_runtime is xsum(total_latency_expr)
    # This is a non-linear constraint if total_runtime is a variable sum.
    # Alternative: total_runtime <= (batch_size * seq_len) / min_throughput
    max_runtime_for_throughput = (C.GKD_BATCH_SIZE * C.MAX_SEQ_LENGTH) / C.TARGET_THROUGHPUT_MIN
    m += xsum(total_latency_expr) <= max_runtime_for_throughput, "min_throughput"

    print("Optimizing MIP problem...")
    status = m.optimize()

    selected_architecture = []
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        print(f"MIP Solution Found! Objective: {m.objective_value}")
        for i in range(num_layers):
            chosen_attn_variant = None
            for j, var_info in enumerate(attn_variants_for_mip[i]):
                if x_attn[i][j].x >= 0.99: # Check if selected
                    chosen_attn_variant = var_info
                    break
            
            chosen_ffn_variant = None
            for j, var_info in enumerate(ffn_variants_for_mip[i]):
                if x_ffn[i][j].x >= 0.99:
                    chosen_ffn_variant = var_info
                    break
            
            if chosen_attn_variant and chosen_ffn_variant:
                selected_architecture.append({
                    'layer_idx': i,
                    'attention_variant': chosen_attn_variant,
                    'ffn_variant': chosen_ffn_variant
                })
            else:
                print(f"Error: No variant selected for layer {i}")
                return None
        return selected_architecture
    else:
        print(f"MIP optimization failed or no solution found. Status: {status}")
        return None