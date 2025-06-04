# local_distillation.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import copy
from tqdm import tqdm
# import math # Not strictly needed unless re-enabling certain debug blocks

from utils import (
    get_device, get_model_layers, set_attention_block, set_ffn_block,
    get_attention_block, get_ffn_block, set_model_layers,
    simulate_block_cost, get_dummy_dataloader, NoOpModule
)
import config as C # This is your puzzle_config object
from block_variants import get_block_alternatives


def train_single_block_variant(parent_model, tokenizer,
                               layer_idx, block_variant_info,
                               original_sibling_block,
                               is_attention_variant):
    if not hasattr(C, 'DEVICE') or C.DEVICE is None:
        C.DEVICE = get_device()
    device = C.DEVICE

    def prepare_attention_mask(raw_attention_mask, device, target_dtype, model_type, seq_length=None):
        """辅助函数：为attention模块准备正确格式的attention mask"""
        if model_type in ["gpt2", "gpt_neo"]:
            batch_size = raw_attention_mask.shape[0]
            seq_length = seq_length or raw_attention_mask.shape[1]
            attention_mask_float = raw_attention_mask.to(dtype=target_dtype)
            causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool), diagonal=1).to(device)
            mask_4d = attention_mask_float.view(batch_size, 1, 1, seq_length)
            mask_4d = mask_4d.expand(-1, 1, seq_length, -1)
            mask_4d = mask_4d.masked_fill(causal_mask, 0.0)
            mask_4d = mask_4d.masked_fill(mask_4d == 0, -10000.0)
            return mask_4d.to(target_dtype)
        else:
            return raw_attention_mask.to(dtype=target_dtype)
        
    parent_model.eval()
    child_block_variant = block_variant_info['variant_obj']
    
    params_to_train = list(child_block_variant.parameters())
    if not params_to_train:
        print(f"  Skipping training for {block_variant_info['name']} as it has no trainable parameters.")
        child_block_variant.eval()
        return child_block_variant

    child_block_variant.train().to(device)
    sibling_block = copy.deepcopy(original_sibling_block).to(device) 
    sibling_block.eval() 

    optimizer = optim.AdamW(params_to_train, lr=C.BLD_LR) # type: ignore
    mse_loss_fn = nn.MSELoss()

    parent_layers_ref = get_model_layers(parent_model)
    original_parent_target_layer = copy.deepcopy(parent_layers_ref[layer_idx]).to(device)
    
    model_type = parent_model.config.model_type # Get model_type once
    # Ensure submodules of the copied layer are also on the device
    if model_type in ["gpt2", "gpt_neo"]:
        if hasattr(original_parent_target_layer, 'attn'): 
            original_parent_target_layer.attn = original_parent_target_layer.attn.to(device)
        if hasattr(original_parent_target_layer, 'mlp'):
            original_parent_target_layer.mlp = original_parent_target_layer.mlp.to(device)
    elif model_type == "opt":
        if hasattr(original_parent_target_layer, 'self_attn'):
             original_parent_target_layer.self_attn = original_parent_target_layer.self_attn.to(device)
        if hasattr(original_parent_target_layer, 'fc1'):
            original_parent_target_layer.fc1 = original_parent_target_layer.fc1.to(device)
        if hasattr(original_parent_target_layer, 'fc2'):
            original_parent_target_layer.fc2 = original_parent_target_layer.fc2.to(device)
    original_parent_target_layer.eval()


    bld_num_samples = C.BLD_VALIDATION_SET_SIZE 
    dataloader = get_dummy_dataloader(tokenizer, C.BLD_BATCH_SIZE, bld_num_samples, C.MAX_SEQ_LENGTH, 
                                      C.DISTILLATION_MIX_PATH, C.DISTILLATION_MIX_SUBSET)

    print(f"BLD for: {block_variant_info['name']} (Layer {layer_idx}, Model type: {model_type})")
    
    target_float_dtype = parent_model.dtype

    for epoch in range(C.BLD_EPOCHS):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            if C.BLD_EPOCHS > 1 and batch_idx * C.BLD_BATCH_SIZE * C.MAX_SEQ_LENGTH > 200000 and C.BLD_BATCH_SIZE > 0:
                print("  (Reached token limit for faster dev in BLD training epoch)")
                break

            input_ids = batch['input_ids'].to(device)
            raw_attention_mask = batch['attention_mask'].to(device)
            attention_mask_preprocessed = prepare_attention_mask(raw_attention_mask, device, target_float_dtype, model_type, seq_length=input_ids.shape[1])

            # 维持对attention_mask_float的引用以用于调试
            attention_mask_float = raw_attention_mask.to(dtype=target_float_dtype)
            
            current_hidden_states = None
            if model_type in ["gpt2", "gpt_neo"]:
                if hasattr(parent_model.transformer, 'wte') and hasattr(parent_model.transformer, 'wpe'):
                    current_hidden_states = parent_model.transformer.wte(input_ids) + \
                                         parent_model.transformer.wpe(torch.arange(input_ids.size(1), device=device))
                else: raise AttributeError(f"wte/wpe not found for {model_type}")
            elif model_type == "opt":
                if hasattr(parent_model.model.decoder, 'embed_tokens') and hasattr(parent_model.model.decoder, 'embed_positions'):
                    inputs_embeds = parent_model.model.decoder.embed_tokens(input_ids)
                    pos_embeds = parent_model.model.decoder.embed_positions(input_ids, past_key_values_length=0)
                    current_hidden_states = inputs_embeds + pos_embeds
                else: raise AttributeError(f"embed_tokens/embed_positions not found for {model_type}")
            else:
                raise NotImplementedError(f"Embedding layer access not implemented for model type {model_type}")
            current_hidden_states = current_hidden_states.to(target_float_dtype)

            for i_prev_layer in range(layer_idx):
                layer_to_use = parent_layers_ref[i_prev_layer].to(device).eval()
                # 使用preprocessed的attention mask
                layer_outputs_tuple = layer_to_use(current_hidden_states, 
                                                 attention_mask=attention_mask_preprocessed,
                                                 use_cache=False)
                if not isinstance(layer_outputs_tuple, tuple) or len(layer_outputs_tuple) == 0:
                    raise TypeError(f"Layer {i_prev_layer} of type {type(layer_to_use)} did not return a non-empty tuple.")
                current_hidden_states = layer_outputs_tuple[0]
            
            target_layer_input_activations = current_hidden_states.detach().clone()

            if batch_idx == 0 and epoch == 0: 
                print(f"\n[BLD Train Loop Debug] Epoch {epoch}, Layer {layer_idx}, Variant {block_variant_info['name']}")
                print(f"  target_layer_input_activations.shape: {target_layer_input_activations.shape}, dtype: {target_layer_input_activations.dtype}")
                print(f"  raw_attention_mask.shape: {raw_attention_mask.shape}, dtype: {raw_attention_mask.dtype}")
                print(f"  attention_mask_float.shape: {attention_mask_float.shape}, dtype: {attention_mask_float.dtype}")

            target_outputs_to_mimic = None
            with torch.no_grad():
                parent_block_to_mimic = None
                if is_attention_variant:
                    parent_block_to_mimic = get_attention_block(original_parent_target_layer)
                    if batch_idx == 0 and epoch == 0 and hasattr(parent_block_to_mimic, 'num_heads'): 
                        print(f"  parent_block_to_mimic type: {type(parent_block_to_mimic)}")
                        print(f"  parent_block_to_mimic num_heads: {parent_block_to_mimic.num_heads}, head_dim: {parent_block_to_mimic.head_dim if hasattr(parent_block_to_mimic, 'head_dim') else 'N/A'}")
                    
                    attention_mask_final = prepare_attention_mask(raw_attention_mask, device, target_float_dtype, model_type, seq_length=input_ids.shape[1])
                    
                    attn_kwargs = {
                        "attention_mask": attention_mask_final,
                        "use_cache": False,
                        "output_attentions": False
                    }
                    if model_type in ["gpt2", "gpt_neo"]: 
                        attn_kwargs["layer_past"] = None
                    elif model_type == "opt": 
                        attn_kwargs["past_key_value"] = None
                    if hasattr(parent_block_to_mimic, 'forward') and 'head_mask' in parent_block_to_mimic.forward.__code__.co_varnames:
                        attn_kwargs["head_mask"] = None
                        
                    parent_block_outputs_tuple = parent_block_to_mimic(target_layer_input_activations, **attn_kwargs)
                    if not isinstance(parent_block_outputs_tuple, tuple) or len(parent_block_outputs_tuple)==0:
                         raise TypeError(f"Parent attention block {type(parent_block_to_mimic)} did not return a non-empty tuple.")
                    target_outputs_to_mimic = parent_block_outputs_tuple[0]
                else: # is_ffn_variant
                    parent_block_to_mimic = get_ffn_block(original_parent_target_layer)
                    original_attn_in_layer = get_attention_block(original_parent_target_layer)
                    
                    # 修正：为FFN mimic分支也使用prepare_attention_mask生成4D mask
                    attn_mask_for_ffn_mimic = prepare_attention_mask(raw_attention_mask, device, target_float_dtype, model_type, seq_length=input_ids.shape[1])
                    attn_kwargs_for_ffn_input = {"attention_mask": attn_mask_for_ffn_mimic, "use_cache": False, "output_attentions": False}
                    if model_type in ["gpt2", "gpt_neo"]: attn_kwargs_for_ffn_input["layer_past"] = None
                    elif model_type == "opt": attn_kwargs_for_ffn_input["past_key_value"] = None
                    if hasattr(original_attn_in_layer, 'forward') and 'head_mask' in original_attn_in_layer.forward.__code__.co_varnames:
                        attn_kwargs_for_ffn_input["head_mask"] = None
                        
                    attn_output_tuple = original_attn_in_layer(target_layer_input_activations, **attn_kwargs_for_ffn_input)
                    if not isinstance(attn_output_tuple, tuple) or len(attn_output_tuple)==0:
                        raise TypeError(f"Original attention block {type(original_attn_in_layer)} in FFN pre-calc did not return a non-empty tuple.")
                    attn_output_for_ffn = attn_output_tuple[0]
                    
                    hidden_states_after_attn_residual = target_layer_input_activations + attn_output_for_ffn
                    
                    ffn_input_actual = None
                    if model_type in ["gpt2", "gpt_neo"]:
                        if hasattr(original_parent_target_layer, 'ln_2'): ffn_input_actual = original_parent_target_layer.ln_2(hidden_states_after_attn_residual)
                        else: raise AttributeError(f"ln_2 not found in {type(original_parent_target_layer)} for {model_type}")
                    elif model_type == "opt":
                        if hasattr(original_parent_target_layer, 'final_layer_norm'): ffn_input_actual = original_parent_target_layer.final_layer_norm(hidden_states_after_attn_residual)
                        else: raise AttributeError(f"final_layer_norm not found in {type(original_parent_target_layer)} for {model_type}")
                    elif hasattr(original_parent_target_layer, 'post_attention_layernorm'):
                         ffn_input_actual = original_parent_target_layer.post_attention_layernorm(hidden_states_after_attn_residual)
                    else: raise NotImplementedError(f"LayerNorm before FFN not found for {type(original_parent_target_layer)} and {model_type}")
                    target_outputs_to_mimic = parent_block_to_mimic(ffn_input_actual)

            optimizer.zero_grad()
            current_variant_output = None
            if is_attention_variant:
                # 使用预处理好的attention mask
                attention_mask_final = attention_mask_preprocessed

                child_attn_kwargs = {
                    "attention_mask": attention_mask_final,
                    "use_cache": False,
                    "output_attentions": False
                }
                if model_type in ["gpt2", "gpt_neo"]: child_attn_kwargs["layer_past"] = None
                elif model_type == "opt": child_attn_kwargs["past_key_value"] = None
                if hasattr(child_block_variant, 'forward') and 'head_mask' in child_block_variant.forward.__code__.co_varnames:
                     child_attn_kwargs["head_mask"] = None
                child_variant_outputs_tuple = child_block_variant(target_layer_input_activations, **child_attn_kwargs)
                if not isinstance(child_variant_outputs_tuple, tuple) or len(child_variant_outputs_tuple)==0:
                    raise TypeError(f"Child attention variant {type(child_block_variant)} did not return a non-empty tuple.")
                current_variant_output = child_variant_outputs_tuple[0] 
            else: # is_ffn_variant
                with torch.no_grad(): 
                    parent_attn_of_layer = get_attention_block(original_parent_target_layer)
                    # 使用统一的mask处理
                    attention_mask_ffn = prepare_attention_mask(raw_attention_mask, device, target_float_dtype, model_type, seq_length=input_ids.shape[1])
                    ffn_input_prep_attn_kwargs = {
                        "attention_mask": attention_mask_ffn,
                        "use_cache": False,
                        "output_attentions": False
                    }
                    if model_type in ["gpt2", "gpt_neo"]: ffn_input_prep_attn_kwargs["layer_past"] = None
                    elif model_type == "opt": ffn_input_prep_attn_kwargs["past_key_value"] = None
                    if hasattr(parent_attn_of_layer, 'forward') and 'head_mask' in parent_attn_of_layer.forward.__code__.co_varnames:
                        ffn_input_prep_attn_kwargs["head_mask"] = None
                    attn_outputs_tuple_for_ffn = parent_attn_of_layer(target_layer_input_activations, **ffn_input_prep_attn_kwargs)
                    if not isinstance(attn_outputs_tuple_for_ffn, tuple) or len(attn_outputs_tuple_for_ffn)==0:
                         raise TypeError(f"Parent attention (for FFN input) {type(parent_attn_of_layer)} did not return a non-empty tuple.")
                    attn_outputs_for_ffn_input_val = attn_outputs_tuple_for_ffn[0]
                    hidden_states_after_attn_residual_val = target_layer_input_activations + attn_outputs_for_ffn_input_val
                    
                    ffn_input_actual_val = None
                    if model_type in ["gpt2", "gpt_neo"]:
                        if hasattr(original_parent_target_layer, 'ln_2'): ffn_input_actual_val = original_parent_target_layer.ln_2(hidden_states_after_attn_residual_val)
                        else: raise AttributeError(f"ln_2 not found in {type(original_parent_target_layer)} for {model_type} during child FFN input prep.")
                    elif model_type == "opt":
                        if hasattr(original_parent_target_layer, 'final_layer_norm'): ffn_input_actual_val = original_parent_target_layer.final_layer_norm(hidden_states_after_attn_residual_val)
                        else: raise AttributeError(f"final_layer_norm not found in {type(original_parent_target_layer)} for {model_type} during child FFN input prep.")
                    elif hasattr(original_parent_target_layer, 'post_attention_layernorm'):
                         ffn_input_actual_val = original_parent_target_layer.post_attention_layernorm(hidden_states_after_attn_residual_val)
                    else: raise NotImplementedError(f"LayerNorm before FFN not found for {type(original_parent_target_layer)} / {model_type} during child FFN input prep.")
                current_variant_output = child_block_variant(ffn_input_actual_val)

            loss = mse_loss_fn(current_variant_output, target_outputs_to_mimic.detach())
            loss.backward()
            optimizer.step()

            if batch_idx > 0 and batch_idx % 50 == 0 and hasattr(dataloader, 'batch_size') and dataloader.batch_size is not None:
                print(f"  Layer {layer_idx}, Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    child_block_variant.eval()
    return child_block_variant

def score_block_variant_kl_divergence(parent_model, tokenizer, layer_idx, 
                                      trained_block_variant, block_type_str,
                                      validation_dataloader):
    if not hasattr(C, 'DEVICE') or C.DEVICE is None:
        C.DEVICE = get_device()
    device = C.DEVICE

    temp_model = copy.deepcopy(parent_model).to(device)
    temp_model.eval()
    parent_model.to(device).eval()

    temp_layers = get_model_layers(temp_model)
    trained_block_variant_on_device = trained_block_variant.to(device)

    target_layer_in_temp_model = temp_layers[layer_idx].to(device)
    if block_type_str == 'attention':
        set_attention_block(target_layer_in_temp_model, trained_block_variant_on_device)
    else: 
        set_ffn_block(target_layer_in_temp_model, trained_block_variant_on_device)
    set_model_layers(temp_model, temp_layers)

    total_kl_div = 0
    num_batches = 0
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True) 

    with torch.no_grad():
        for batch_idx_score, batch in enumerate(tqdm(validation_dataloader, desc=f"Scoring KL Div L{layer_idx} {block_type_str} {trained_block_variant.__class__.__name__}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask_from_loader = batch['attention_mask'].to(device) 

            if batch_idx_score == 0:
                print(f"\n[KL Score Debug] Layer {layer_idx}, Variant {trained_block_variant.__class__.__name__}, Type {block_type_str}")
                print(f"  input_ids.shape: {input_ids.shape}, dtype: {input_ids.dtype}")
                print(f"  attention_mask_from_loader.shape: {attention_mask_from_loader.shape}, dtype: {attention_mask_from_loader.dtype}")

            parent_outputs = parent_model(input_ids, attention_mask=attention_mask_from_loader)
            child_outputs = temp_model(input_ids, attention_mask=attention_mask_from_loader)

            parent_logits = parent_outputs.logits
            child_logits = child_outputs.logits
            
            parent_log_probs = F.log_softmax(parent_logits.float(), dim=-1)
            child_log_probs = F.log_softmax(child_logits.float(), dim=-1)
            
            kl_div = kl_loss_fn(child_log_probs, parent_log_probs)
            total_kl_div += kl_div.item()
            num_batches += 1
            
            if num_batches >= 20 and hasattr(validation_dataloader, 'dataset') and len(validation_dataloader.dataset) > 100:
                 break

    avg_kl_div = total_kl_div / num_batches if num_batches > 0 else float('inf')
    print(f"  Avg KL Div for layer {layer_idx} ({block_type_str}, {trained_block_variant.__class__.__name__}): {avg_kl_div:.4f}")
    return avg_kl_div

def run_bld_and_scoring(parent_model, tokenizer):
    if not hasattr(C, 'DEVICE') or C.DEVICE is None:
        C.DEVICE = get_device()
    device = C.DEVICE 

    parent_model.to(device) 
    parent_layers_orig = get_model_layers(parent_model)
    num_layers = len(parent_layers_orig)
    block_library = [[] for _ in range(num_layers)]
    
    scoring_batch_size = C.BLD_BATCH_SIZE // 2 if C.BLD_BATCH_SIZE > 1 else 1
    validation_set_size_for_scoring = max(10, C.BLD_VALIDATION_SET_SIZE) 
    scoring_dataloader = get_dummy_dataloader(tokenizer, scoring_batch_size, 
                                              validation_set_size_for_scoring, C.MAX_SEQ_LENGTH, 
                                              C.DISTILLATION_MIX_PATH, C.DISTILLATION_MIX_SUBSET)

    for i in range(num_layers):
        print(f"\nProcessing Layer {i}/{num_layers-1}")
        parent_layer_i_original = copy.deepcopy(parent_layers_orig[i]).to(device)
        model_type = parent_model.config.model_type
        if model_type in ["gpt2", "gpt_neo"]:
            if hasattr(parent_layer_i_original, 'attn'): 
                parent_layer_i_original.attn = parent_layer_i_original.attn.to(device)
            if hasattr(parent_layer_i_original, 'mlp'):
                parent_layer_i_original.mlp = parent_layer_i_original.mlp.to(device)
        elif model_type == "opt":
            if hasattr(parent_layer_i_original, 'self_attn'):
                 parent_layer_i_original.self_attn = parent_layer_i_original.self_attn.to(device)
            if hasattr(parent_layer_i_original, 'fc1'):
                parent_layer_i_original.fc1 = parent_layer_i_original.fc1.to(device)
            if hasattr(parent_layer_i_original, 'fc2'):
                parent_layer_i_original.fc2 = parent_layer_i_original.fc2.to(device)

        alternatives_for_layer_i = get_block_alternatives(parent_layer_i_original, i, C, parent_model.config) 

        for alt_info in alternatives_for_layer_i:
            variant_type = alt_info['type'] 
            variant_name = alt_info['name']

            original_sibling_block_ref = None
            is_attention_variant_training = False
            if variant_type == 'attention':
                original_sibling_block_ref = get_ffn_block(parent_layer_i_original)
                is_attention_variant_training = True
            else: 
                original_sibling_block_ref = get_attention_block(parent_layer_i_original)
                is_attention_variant_training = False
            
            trained_variant_module = train_single_block_variant(
                parent_model, tokenizer, i, alt_info,
                original_sibling_block_ref, is_attention_variant_training
            )
            
            score = score_block_variant_kl_divergence(parent_model, tokenizer, i, 
                                                      trained_variant_module, variant_type,
                                                      scoring_dataloader)
            
            parent_block_for_cost_ref = get_attention_block(parent_layer_i_original) if variant_type == 'attention' else get_ffn_block(parent_layer_i_original)
            cost_metrics = simulate_block_cost(trained_variant_module, parent_block_for_cost_ref,
                                               variant_type, C)

            block_library[i].append({
                'name': variant_name, 'type': variant_type,
                'trained_obj': trained_variant_module.cpu(), 'score': score,
                'cost_latency': cost_metrics['latency'], 'cost_param_memory': cost_metrics['param_memory'],
                'cost_kv_cache_factor': cost_metrics['kv_cache_factor'] 
            })
            
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return block_library