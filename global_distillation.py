import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import copy
from utils import (get_device, get_model_layers, set_model_layers,
                   set_attention_block, set_ffn_block, get_dummy_dataloader)
import config as C

def assemble_child_model(parent_model_architecture, selected_mip_architecture):
    child_model = copy.deepcopy(parent_model_architecture) # Start with parent structure
    child_model.to(C.DEVICE)
    child_layers = get_model_layers(child_model)

    for layer_config in selected_mip_architecture:
        layer_idx = layer_config['layer_idx']
        
        # Get the actual nn.Module objects (they were stored on CPU)
        attn_module = layer_config['attention_variant']['trained_obj'].to(C.DEVICE)
        ffn_module = layer_config['ffn_variant']['trained_obj'].to(C.DEVICE)
        
        set_attention_block(child_layers[layer_idx], attn_module)
        set_ffn_block(child_layers[layer_idx], ffn_module)
        
    set_model_layers(child_model, child_layers)
    return child_model

def gkd_loss_fn(student_logits, teacher_logits, 
                student_hidden_states_all_layers, teacher_hidden_states_all_layers, 
                config):
    loss = 0.0
    
    # KLD Loss for logits
    kld_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)(
        F.log_softmax(student_logits, dim=-1),
        F.log_softmax(teacher_logits, dim=-1)
    )
    loss += config.KLD_WEIGHT * kld_loss
    
    # Cosine Similarity Loss for hidden states (layer-wise)
    # Assuming student_hidden_states and teacher_hidden_states are lists of tensors (one per layer)
    # This requires model to output hidden states: output_hidden_states=True
    num_distill_layers = min(len(student_hidden_states_all_layers), len(teacher_hidden_states_all_layers))
    
    # Skip embedding layer (index 0) if present
    start_idx = 1 if student_hidden_states_all_layers[0].shape == teacher_hidden_states_all_layers[0].shape else 0


    total_cosine_sim_loss = 0
    actual_layers_compared = 0
    for i in range(start_idx, num_distill_layers):
        s_h = student_hidden_states_all_layers[i]
        t_h = teacher_hidden_states_all_layers[i].detach() # Detach teacher
        if s_h.shape == t_h.shape: # Ensure dimensions match if some layers were skipped/modified
            # Normalize before cosine similarity calculation
            s_h_norm = F.normalize(s_h, p=2, dim=-1)
            t_h_norm = F.normalize(t_h, p=2, dim=-1)
            # Cosine similarity is sum(s_h_norm * t_h_norm), loss is 1 - sim or -sim
            # Maximize cosine similarity -> Minimize -cosine_similarity
            # Taking mean over batch and sequence length
            cos_sim_layer = (s_h_norm * t_h_norm).sum(dim=-1).mean()
            total_cosine_sim_loss -= cos_sim_layer # Minimize negative similarity
            actual_layers_compared +=1
        else:
            print(f"Skipping cosine sim for layer {i} due to shape mismatch: S{s_h.shape} vs T{t_h.shape}")


    if actual_layers_compared > 0:
        avg_cosine_sim_loss = total_cosine_sim_loss / actual_layers_compared
        loss += config.COSINE_SIM_WEIGHT * avg_cosine_sim_loss
    
    return loss

def run_gkd(parent_model, child_model, tokenizer):
    parent_model.eval().to(C.DEVICE)
    child_model.train().to(C.DEVICE)

    optimizer = optim.AdamW(child_model.parameters(), lr=C.GKD_LR) # type: ignore
    
    # Using a smaller dataset for GKD example
    num_tokens_processed = 0
    epoch = 0
    
    while num_tokens_processed < C.GKD_NUM_TOKENS:
        epoch += 1
        print(f"GKD Epoch {epoch}")
        # Recreate dataloader each epoch if dataset is small, or use iterable dataset
        gkd_dataloader = get_dummy_dataloader(tokenizer, C.GKD_BATCH_SIZE, 
                                              C.GKD_NUM_TOKENS // (C.MAX_SEQ_LENGTH * C.GKD_BATCH_SIZE) + 10, # num batches
                                              C.MAX_SEQ_LENGTH, C.DISTILLATION_MIX_PATH, C.DISTILLATION_MIX_SUBSET)

        for batch_idx, batch in enumerate(tqdm(gkd_dataloader, desc=f"GKD Epoch {epoch}")):
            input_ids = batch['input_ids'].to(C.DEVICE)
            attention_mask = batch['attention_mask'].to(C.DEVICE)

            optimizer.zero_grad()

            with torch.no_grad():
                parent_outputs = parent_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            child_outputs = child_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

            loss = gkd_loss_fn(child_outputs.logits, parent_outputs.logits,
                               child_outputs.hidden_states, parent_outputs.hidden_states, C)
            
            loss.backward()
            optimizer.step()
            
            num_tokens_processed += input_ids.numel()

            if batch_idx % 50 == 0:
                print(f"  GKD Batch {batch_idx}, Loss: {loss.item()}, Tokens: {num_tokens_processed}/{C.GKD_NUM_TOKENS}")
            
            if num_tokens_processed >= C.GKD_NUM_TOKENS:
                break
    
    child_model.eval()
    return child_model