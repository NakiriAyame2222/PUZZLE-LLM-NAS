import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM # Keep AutoModelForCausalLM for parent loading
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset # type: ignore # For get_dummy_dataloader

# Keep config import if it's used for any constants here, otherwise remove if not needed directly
# import config as C # Example: if C.SOME_UTIL_PARAM was used

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_parent_model_and_tokenizer(model_name, model_dtype=torch.float32): # Add model_dtype
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model and ensure it's in the specified dtype and on the correct device
    parent_model = AutoModelForCausalLM.from_pretrained(model_name).to(get_device()).to(model_dtype).eval()
    print(f"Parent model '{model_name}' loaded with dtype {parent_model.dtype} on device {parent_model.device}.")
    return parent_model, tokenizer

def get_model_layers(model):
    model_type = model.config.model_type
    if model_type in ["gpt2", "gpt_neo"]:
        return model.transformer.h
    elif model_type == "opt":
        return model.model.decoder.layers
    else:
        raise NotImplementedError(f"Layer access not implemented for {model_type}")

def set_model_layers(model, layers_module_list):
    model_type = model.config.model_type
    if model_type in ["gpt2", "gpt_neo"]:
        model.transformer.h = layers_module_list
    elif model_type == "opt":
        model.model.decoder.layers = layers_module_list
    else:
        raise NotImplementedError(f"Layer setting not implemented for {model_type}")

def get_attention_block(layer):
    # Use config.model_type from the layer's config if available, or infer
    layer_model_type = None
    if hasattr(layer, 'config') and hasattr(layer.config, 'model_type'): # For OPTLayer etc.
        layer_model_type = layer.config.model_type
    
    # Fallback to class name inference if config is not directly on layer
    if layer_model_type is None:
        class_name_lower = layer.__class__.__name__.lower()
        if "gpt2" in class_name_lower: layer_model_type = "gpt2"
        elif "gptneo" in class_name_lower: layer_model_type = "gpt_neo"
        elif "opt" in class_name_lower: layer_model_type = "opt"


    if layer_model_type == "gpt2": # GPT2Block
        return layer.attn
    elif layer_model_type == "gpt_neo": # GPTNeoBlock
        return layer.attn.attention # GPTNeoSelfAttention is nested
    elif layer_model_type == "opt": # OPTDecoderLayer
        return layer.self_attn
    raise NotImplementedError(f"Attention block access not implemented for layer type: {type(layer)} (inferred type: {layer_model_type})")

def set_attention_block(layer, new_attn):
    layer_model_type = None
    if hasattr(layer, 'config') and hasattr(layer.config, 'model_type'):
        layer_model_type = layer.config.model_type
    if layer_model_type is None:
        class_name_lower = layer.__class__.__name__.lower()
        if "gpt2" in class_name_lower: layer_model_type = "gpt2"
        elif "gptneo" in class_name_lower: layer_model_type = "gpt_neo"
        elif "opt" in class_name_lower: layer_model_type = "opt"

    if layer_model_type == "gpt2":
        layer.attn = new_attn
    elif layer_model_type == "gpt_neo":
        layer.attn.attention = new_attn 
    elif layer_model_type == "opt":
        layer.self_attn = new_attn
    else:
        raise NotImplementedError(f"Attention block setting not implemented for layer type: {type(layer)} (inferred type: {layer_model_type})")

class OPTFFNWrapper(nn.Module):
    def __init__(self, fc1, fc2, activation_fn_obj):
        super().__init__()
        self.fc1 = fc1
        self.activation_fn = activation_fn_obj 
        self.fc2 = fc2

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

def get_ffn_block(layer):
    layer_model_type = None
    if hasattr(layer, 'config') and hasattr(layer.config, 'model_type'):
        layer_model_type = layer.config.model_type
    if layer_model_type is None:
        class_name_lower = layer.__class__.__name__.lower()
        if "gpt2" in class_name_lower: layer_model_type = "gpt2"
        elif "gptneo" in class_name_lower: layer_model_type = "gpt_neo"
        elif "opt" in class_name_lower: layer_model_type = "opt"

    if layer_model_type == "gpt2":
        return layer.mlp
    elif layer_model_type == "gpt_neo": 
        return layer.mlp 
    elif layer_model_type == "opt": 
        if not hasattr(layer, 'activation_fn') or not callable(layer.activation_fn):
            # OPT's activation_fn string needs to be mapped to an actual function
            # This usually happens inside OPTDecoderLayer's __init__
            # For safety, if it's a string, try to get it. This is a fallback.
            if isinstance(layer.activation_fn, str):
                if layer.activation_fn == "relu": act_fn_obj = F.relu
                elif layer.activation_fn == "gelu": act_fn_obj = F.gelu # or specific gelu_new etc.
                else: raise ValueError(f"Unknown activation string {layer.activation_fn} for OPT")
            else: # Should be the function object itself
                act_fn_obj = layer.activation_fn
        else: # It's already the function object
            act_fn_obj = layer.activation_fn
        return OPTFFNWrapper(layer.fc1, layer.fc2, act_fn_obj)
    raise NotImplementedError(f"FFN block access not implemented for layer type: {type(layer)} (inferred type: {layer_model_type})")

def set_ffn_block(layer, new_ffn):
    layer_model_type = None
    if hasattr(layer, 'config') and hasattr(layer.config, 'model_type'):
        layer_model_type = layer.config.model_type
    if layer_model_type is None:
        class_name_lower = layer.__class__.__name__.lower()
        if "gpt2" in class_name_lower: layer_model_type = "gpt2"
        elif "gptneo" in class_name_lower: layer_model_type = "gpt_neo"
        elif "opt" in class_name_lower: layer_model_type = "opt"

    if layer_model_type == "gpt2":
        layer.mlp = new_ffn
    elif layer_model_type == "gpt_neo":
        layer.mlp = new_ffn
    elif layer_model_type == "opt":
        if isinstance(new_ffn, OPTFFNWrapper):
            layer.fc1 = new_ffn.fc1
            layer.fc2 = new_ffn.fc2
            layer.activation_fn = new_ffn.activation_fn 
        elif isinstance(new_ffn, NoOpModule):
            layer.fc1 = nn.Identity() 
            layer.fc2 = nn.Identity() 
            layer.activation_fn = nn.Identity() # Or lambda x: x
            print(f"  Replaced OPT FFN with NoOp (fc1, fc2, act made Identity).")
        elif isinstance(new_ffn, LinearReplacement):
            layer.fc1 = new_ffn # Replace fc1 with the linear layer
            layer.activation_fn = nn.Identity() # Bypass activation
            layer.fc2 = nn.Identity() # Bypass second linear layer
            print(f"  Replaced OPT fc1 with LinearReplacement, act and fc2 made Identity.")
        else:
            raise ValueError(f"Cannot set FFN for OPT with type {type(new_ffn)}. Expected OPTFFNWrapper, NoOpModule, or LinearReplacement.")
    else:
        raise NotImplementedError(f"FFN block setting not implemented for layer type: {type(layer)} (inferred type: {layer_model_type})")

class NoOpModule(nn.Module):
    def __init__(self, is_attention_replacement=False):
        super().__init__()
        self.is_attention_replacement = is_attention_replacement

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False, past_key_value=None, # Add past_key_value for OPT
                **other_kwargs): # Use other_kwargs for any unexpected ones
        
        if self.is_attention_replacement:
            final_outputs = (x, None) # (main_output, placeholder_for_past_kv)
            if output_attentions: # Check the actual flag passed in the call
                final_outputs = final_outputs + (None,) # placeholder_for_attn_weights
            return final_outputs
        else:
            return x

class LinearReplacement(nn.Module):
    def __init__(self, d_model, is_attention_replacement=False):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.is_attention_replacement = is_attention_replacement

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False, past_key_value=None, # Add past_key_value for OPT
                **other_kwargs):
        lin_out = self.linear(x)
        
        if self.is_attention_replacement:
            final_outputs = (lin_out, None)
            if output_attentions:
                final_outputs = final_outputs + (None,)
            return final_outputs
        else:
            return lin_out

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        count = 0
        for text in texts:
            if text and text.strip(): 
                encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                         padding="max_length", return_tensors="pt")
                self.encodings.append(encoded)
                count +=1
            if count >= 2000: # Limit number of actual samples to speed up dataloader creation
                # print(f"DEBUG: TextDataset created with {len(self.encodings)} samples (capped at 2000).")
                break
        if not self.encodings:
            dummy_text = "This is a dummy sentence for empty dataset."
            encoded = self.tokenizer(dummy_text, truncation=True, max_length=self.max_length, 
                                     padding="max_length", return_tensors="pt")
            self.encodings.append(encoded)
            print("DEBUG: TextDataset was empty, added one dummy sentence.")


    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.encodings[idx].items()}
        item['labels'] = item['input_ids'].clone()
        return item

def get_dummy_dataloader(tokenizer, batch_size, num_samples_to_fetch, seq_length, 
                         dataset_name="wikitext", subset="wikitext-2-raw-v1"):
    # num_samples_to_fetch is how many samples the dataset object will try to load initially.
    # The TextDataset itself might cap this further for speed.
    print(f"Attempting to load {num_samples_to_fetch} samples for dataloader from {dataset_name}/{subset}...")
    try:
        # Ensure num_samples_to_fetch is at least 1 for split argument
        effective_num_samples = max(1, num_samples_to_fetch)
        dataset = load_dataset(dataset_name, subset, split=f'train[:{effective_num_samples}]', trust_remote_code=True)
        texts = [item['text'] for item in dataset if item['text'] and item['text'].strip()]
        if not texts: # If all fetched items were empty/None
            raise ValueError("Fetched dataset samples were all empty.")
    except Exception as e:
        print(f"Failed to load {dataset_name}/{subset} (or all samples were empty): {e}. Using dummy data strings.")
        texts = [f"This is dummy sentence number {i+1} for the dataloader." for i in range(num_samples_to_fetch if num_samples_to_fetch > 0 else 2)]
        if not texts: texts = ["Dummy fallback sentence 1.", "Dummy fallback sentence 2."]
    
    # TextDataset will internally cap to 2000 actual encoded samples for speed
    custom_dataset = TextDataset(texts, tokenizer, seq_length)
    actual_dataloader_size = len(custom_dataset)
    print(f"DataLoader created with {actual_dataloader_size} samples (batch size {batch_size}).")
    
    if actual_dataloader_size == 0 :
        raise ValueError("TextDataset resulted in zero encodings. Cannot create DataLoader.")
    if batch_size == 0:
        raise ValueError("Batch size for DataLoader cannot be zero.")
    if actual_dataloader_size < batch_size and actual_dataloader_size > 0:
        print(f"Warning: Number of samples ({actual_dataloader_size}) is less than batch_size ({batch_size}). Setting batch_size to {actual_dataloader_size}.")
        batch_size = actual_dataloader_size
        
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=True) # Shuffle for better training


def simulate_block_cost(block_variant, parent_block, block_type, puzzle_config_obj): # Renamed config to avoid clash
    latency_cost = 0
    memory_cost = 0 
    kv_cache_factor = 0 

    if isinstance(block_variant, NoOpModule):
        latency_cost = 0.1 
        memory_cost = 0.001
    elif isinstance(block_variant, LinearReplacement):
        base_lat = puzzle_config_obj.BASE_ATTENTION_COST if block_type == 'attention' else puzzle_config_obj.BASE_FFN_COST
        base_mem = puzzle_config_obj.BASE_ATTENTION_MEMORY if block_type == 'attention' else puzzle_config_obj.BASE_FFN_MEMORY
        latency_cost = base_lat * 0.3
        memory_cost = base_mem * 0.3
        if block_type == 'attention': kv_cache_factor = 0.1 
    elif block_type == 'attention':
        latency_cost = puzzle_config_obj.BASE_ATTENTION_COST
        memory_cost = puzzle_config_obj.BASE_ATTENTION_MEMORY
        kv_cache_factor = 1.0 
    elif block_type == 'ffn':
        latency_cost = puzzle_config_obj.BASE_FFN_COST
        memory_cost = puzzle_config_obj.BASE_FFN_MEMORY
    
    # TODO: Add more sophisticated cost estimation based on block_variant's actual parameters
    # e.g., for reduced FFN, scale cost by reduction_factor. For GQA, by num_kv_heads.

    return {
        "latency": latency_cost, 
        "param_memory": memory_cost, 
        "kv_cache_factor": kv_cache_factor 
    }