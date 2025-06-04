import torch
from utils import load_parent_model_and_tokenizer
from local_distillation import run_bld_and_scoring
# from architecture_search import solve_mip # Comment out or remove MIP
from rl_architecture_search import run_rl_search # Import RL search
from global_distillation import assemble_child_model, run_gkd
import config as C

def main():
    print(f"Using device: {C.DEVICE}")
    parent_model, tokenizer = load_parent_model_and_tokenizer(C.PARENT_MODEL_NAME)
    print(f"Parent model '{C.PARENT_MODEL_NAME}' loaded.")

    print("\n--- Starting Stage 1: BLD and Scoring ---")
    block_library = run_bld_and_scoring(parent_model, tokenizer)
    
    if not any(block_library) or not all(block_library): # Check if library or any layer in it is empty
        print("Block library is empty or incomplete. Exiting.")
        return

    # --- Stage 2: RL-based Architecture Search ---
    print("\n--- Starting Stage 2: RL Architecture Search ---")
    # selected_architecture will be a list of layer_configs:
    # [{'layer_idx', 'attention_variant_info', 'ffn_variant_info'}, ...]
    selected_architecture_rl = run_rl_search(block_library, parent_model.config)

    if not selected_architecture_rl:
        print("RL search did not find a solution. Exiting.")
        return
    
    print("\nSelected Architecture by RL:")
    for i, layer_sel in enumerate(selected_architecture_rl):
        print(f"  Layer {i}: Attn='{layer_sel['attention_variant']['name']}', FFN='{layer_sel['ffn_variant']['name']}'")

    # --- Stage 3: Global Knowledge Distillation (Uptraining) ---
    print("\n--- Starting Stage 3: Global Knowledge Distillation ---")
    initial_child_model_rl = assemble_child_model(parent_model, selected_architecture_rl)
    print("Child model assembled from RL solution.")

    final_child_model_rl = run_gkd(parent_model, initial_child_model_rl, tokenizer)
    print("GKD finished for RL-derived model.")

    # === save the final child model ===
    save_path = "final_child_model_rl.pt"
    torch.save(final_child_model_rl.state_dict(), save_path)
    print(f"model has been saved in {save_path}")

    print("\nPUZZLE framework with RL search conceptual run complete.")

if __name__ == "__main__":
    if not hasattr(C, 'DEVICE'):
         C.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()