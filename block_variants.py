# block_variants.py

import torch
import torch.nn as nn
import copy
from utils import NoOpModule, LinearReplacement, get_attention_block, get_ffn_block
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
# from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP # If using Llama

def create_ffn_variant(parent_ffn_block, reduction_factor, config, d_model, parent_config):
    if reduction_factor is None: 
        return copy.deepcopy(parent_ffn_block)
    
    if isinstance(parent_ffn_block, GPT2MLP):
        # For GPT2MLP:
        # self.c_fc = Conv1D(intermediate_size, embed_dim) -> weight (embed_dim, intermediate_size)
        # self.c_proj = Conv1D(embed_dim, intermediate_size) -> weight (intermediate_size, embed_dim)
        
        original_intermediate_size = parent_ffn_block.c_fc.nf # nf is output of c_fc, input of c_proj
        new_intermediate_size = int(original_intermediate_size * reduction_factor)
        
        new_ffn = GPT2MLP(new_intermediate_size, parent_config)

        # parent_ffn_block.c_fc.weight shape: (d_model, original_intermediate_size)
        # new_ffn.c_fc.weight target shape: (d_model, new_intermediate_size)
        source_slice_c_fc_w = parent_ffn_block.c_fc.weight.data[:, :new_intermediate_size]
        if new_ffn.c_fc.weight.shape == source_slice_c_fc_w.shape:
            new_ffn.c_fc.weight.data = source_slice_c_fc_w
        else:
            print(f"Warning: Shape mismatch for c_fc weight. Target: {new_ffn.c_fc.weight.shape}, Source slice: {source_slice_c_fc_w.shape}. Skipping weight copy.")
        
        # parent_ffn_block.c_fc.bias shape: (original_intermediate_size,)
        # new_ffn.c_fc.bias target shape: (new_intermediate_size,)
        source_slice_c_fc_b = parent_ffn_block.c_fc.bias.data[:new_intermediate_size]
        if new_ffn.c_fc.bias.shape == source_slice_c_fc_b.shape:
            new_ffn.c_fc.bias.data = source_slice_c_fc_b
        else:
            print(f"Warning: Shape mismatch for c_fc bias. Target: {new_ffn.c_fc.bias.shape}, Source slice: {source_slice_c_fc_b.shape}. Skipping bias copy.")

        # parent_ffn_block.c_proj.weight shape: (original_intermediate_size, d_model)
        # new_ffn.c_proj.weight target shape: (new_intermediate_size, d_model)
        source_slice_c_proj_w = parent_ffn_block.c_proj.weight.data[:new_intermediate_size, :]
        if new_ffn.c_proj.weight.shape == source_slice_c_proj_w.shape:
            new_ffn.c_proj.weight.data = source_slice_c_proj_w
        else:
            print(f"Warning: Shape mismatch for c_proj weight. Target: {new_ffn.c_proj.weight.shape}, Source slice: {source_slice_c_proj_w.shape}. Skipping weight copy.")
        
        # parent_ffn_block.c_proj.bias shape: (d_model,) - this is the final output bias of FFN
        # new_ffn.c_proj.bias target shape: (d_model,)
        if new_ffn.c_proj.bias.shape == parent_ffn_block.c_proj.bias.data.shape:
            new_ffn.c_proj.bias.data = parent_ffn_block.c_proj.bias.data
        else:
             print(f"Warning: Shape mismatch for c_proj bias. Target: {new_ffn.c_proj.bias.shape}, Source: {parent_ffn_block.c_proj.bias.data.shape}. Skipping bias copy.")
        return new_ffn
    
    raise NotImplementedError(f"FFN variant creation not implemented for this block type: {type(parent_ffn_block)}")

def create_attention_variant(parent_attn_block, num_kv_heads_option, config, d_model):
    """
    Creates an Attention variant.
    Currently, GQA implementation is a placeholder and returns a copy.
    """
    if num_kv_heads_option is None: # Original
        return copy.deepcopy(parent_attn_block)

    # Placeholder for actual GQA implementation.
    # This would involve modifying projection matrices (k_proj, v_proj) and attention logic.
    print(f"Warning: GQA variant for {num_kv_heads_option} KV heads is complex and not fully implemented, returning copy of parent attention.")
    new_attn = copy.deepcopy(parent_attn_block)
    # Example of what might be needed for GQA (highly model-dependent):
    # if hasattr(new_attn, 'num_key_value_heads'):
    #     new_attn.num_key_value_heads = num_kv_heads_option
    #     # Adjust k_proj, v_proj dimensions and potentially q_proj if head_dim changes.
    #     # This is non-trivial and specific to the attention implementation (e.g., LlamaAttention).
    return new_attn


def get_block_alternatives(parent_layer, layer_idx, config, parent_model_config):
    alternatives = []
    device = config.DEVICE 
    d_model = parent_model_config.hidden_size if hasattr(parent_model_config, 'hidden_size') else parent_model_config.n_embd

    parent_attn = get_attention_block(parent_layer)
    parent_ffn = get_ffn_block(parent_layer)

    # --- Attention Alternatives ---
    if config.ATTENTION_VARIANTS_CONFIG.get("no_op_attention", False):
        # Pass is_attention_replacement=True
        alternatives.append({'type': 'attention', 
                             'variant_obj': NoOpModule(is_attention_replacement=True).to(device), 
                             'name': f'att_noop_L{layer_idx}'})
    if config.ATTENTION_VARIANTS_CONFIG.get("use_linear_attention", False):
        # Pass is_attention_replacement=True
        alternatives.append({'type': 'attention', 
                             'variant_obj': LinearReplacement(d_model, is_attention_replacement=True).to(device), 
                             'name': f'att_linear_L{layer_idx}'})
    
    original_attn_added = False
    # ... (loop for GQA, if create_attention_variant also gets the flag or handles tuple return) ...
    # For GQA variants created by create_attention_variant, that function would also need to ensure tuple returns if it's a full replacement.
    # If it's just modifying the original parent_attn, then parent_attn already handles tuples.
    # For now, assuming create_attention_variant returns a module that behaves like original if it's a copy.
    for n_kv in config.ATTENTION_VARIANTS_CONFIG.get("num_kv_heads_options", []):
        if n_kv is None: 
            if not original_attn_added: 
                 # Original attention block already returns tuples correctly
                 alternatives.append({'type': 'attention', 'variant_obj': copy.deepcopy(parent_attn).to(device), 'name': f'att_orig_L{layer_idx}'})
                 original_attn_added = True
        else: 
            attn_var = create_attention_variant(parent_attn, n_kv, config, d_model) # This needs to be tuple-aware if it's a full custom module
            alternatives.append({'type': 'attention', 'variant_obj': attn_var.to(device), 'name': f'att_gqa{n_kv}_L{layer_idx}'})
    
    if not original_attn_added: 
        alternatives.append({'type': 'attention', 'variant_obj': copy.deepcopy(parent_attn).to(device), 'name': f'att_orig_L{layer_idx}'})


    # --- FFN Alternatives ---
    # For FFNs, is_attention_replacement is False (default)
    if config.FFN_VARIANTS_CONFIG.get("no_op_ffn", False):
        alternatives.append({'type': 'ffn', 
                             'variant_obj': NoOpModule(is_attention_replacement=False).to(device), 
                             'name': f'ffn_noop_L{layer_idx}'})
    if config.FFN_VARIANTS_CONFIG.get("use_linear_ffn", False):
        alternatives.append({'type': 'ffn', 
                             'variant_obj': LinearReplacement(d_model, is_attention_replacement=False).to(device), 
                             'name': f'ffn_linear_L{layer_idx}'})
    # ... (loop for FFN reduction factors, original FFN already returns Tensor) ...
    original_ffn_added = False
    for factor in config.FFN_VARIANTS_CONFIG.get("intermediate_reduction_factors", []):
        if factor is None: 
            if not original_ffn_added:
                # Original FFN block already returns a Tensor correctly
                alternatives.append({'type': 'ffn', 'variant_obj': copy.deepcopy(parent_ffn).to(device), 'name': f'ffn_orig_L{layer_idx}'})
                original_ffn_added = True
        else: 
            ffn_var = create_ffn_variant(parent_ffn, factor, config, d_model, parent_model_config)
            alternatives.append({'type': 'ffn', 'variant_obj': ffn_var.to(device), 'name': f'ffn_reduct{factor*100:.0f}_L{layer_idx}'})
            
    if not original_ffn_added: 
        alternatives.append({'type': 'ffn', 'variant_obj': copy.deepcopy(parent_ffn).to(device), 'name': f'ffn_orig_L{layer_idx}'})
            
    return alternatives