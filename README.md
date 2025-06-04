
# PUZZLE-LLM-NAS: Conceptual Implementation

This project provides a conceptual Python implementation of the PUZZLE framework, a distillation-based Neural Architecture Search (NAS) technique for creating inference-optimized Large Language Models (LLMs), as described in the paper "PUZZLE: DISTILLATION-BASED NAS FOR INFERENCE-OPTIMIZED LLMS". It also draws inspiration from latency-aware acceleration techniques like LANA.

The goal of this project is to outline the algorithmic flow of PUZZLE, allowing for experimentation with different parent models and architectural search spaces on a smaller, more manageable scale than the original research.

## Framework Overview

The PUZZLE framework implemented here follows a three-stage process:

1.  **Stage 1: Block-wise Local Distillation (BLD) & Scoring (`local_distillation.py`)**
    *   A "parent" LLM is loaded.
    *   For each layer of the parent model, alternative Attention and FFN (Feed-Forward Network) sub-blocks are defined (e.g., original, reduced complexity, no-op, linear replacement).
    *   Each alternative sub-block варіант is trained independently (decoupled BLD) to mimic the output of its corresponding sub-block in the parent layer, using a small distillation dataset.
    *   After local distillation, each trained block variant is scored based on its impact on the full model's performance (e.g., using KL divergence against the parent model's outputs on a validation set).
    *   Inference costs (latency, memory) for each block variant are simulated.

2.  **Stage 2: Architecture Search (`architecture_search.py` or `rl_architecture_search.py`)**
    *   Using the library of trained and scored block variants from Stage 1.
    *   An optimization algorithm (either Mixed-Integer Programming (MIP) or Reinforcement Learning (RL) - select one in `main.py`) searches for the optimal combination of Attention and FFN blocks for each layer.
    *   The objective is to maximize (or minimize, e.g., KL divergence) the sum of block quality scores while adhering to specified inference constraints (e.g., target throughput, latency, memory).

3.  **Stage 3: Global Knowledge Distillation (GKD) (`global_distillation.py`)**
    *   The child LLM architecture, as determined by the search algorithm in Stage 2, is assembled.
    *   This child model is then "uptrained" using global knowledge distillation. The parent model acts as the teacher.
    *   The GKD loss typically combines a KLDivergence loss on the output logits and a Cosine Similarity loss on hidden state representations between the child and parent models.

## Project Structure

```
puzzle-llm-nas/
├── main.py                     # Main script to run the PUZZLE framework
├── config.py                   # Configuration for models, training, search, etc.
├── utils.py                    # Utility functions (model loading, layer access, custom modules)
├── block_variants.py           # Defines and creates alternative Attention/FFN blocks
├── local_distillation.py       # Implements Stage 1: BLD and scoring
├── architecture_search.py      # Implements Stage 2: MIP-based architecture search
├── rl_architecture_search.py   # Implements Stage 2: RL-based architecture search (alternative)
├── global_distillation.py      # Implements Stage 3: GKD
└── README.md                   # This file
```

## Prerequisites

*   Python 3.8+
*   PyTorch
*   Hugging Face Transformers
*   Hugging Face Datasets
*   `mip` (for MIP-based search): `pip install mip`
*   `tqdm` (for progress bars)
*   `accelerate` (often a dependency for Transformers)

Install dependencies using pip:
```bash
pip install torch transformers datasets mip tqdm accelerate
```

## Usage

The main script to run the entire PUZZLE framework is `main.py`.

### Running the Framework

1.  **Configure `config.py`**:
    *   Set `PARENT_MODEL_NAME` to the Hugging Face model identifier of the parent LLM you want to optimize (e.g., "gpt2", "distilgpt2", "EleutherAI/gpt-neo-125M").
    *   Adjust `BLD_EPOCHS`, `GKD_EPOCHS`, batch sizes, learning rates, etc., as needed for your experiments and available resources.
    *   Configure `TARGET_THROUGHPUT_MIN`, `TARGET_LATENCY_MAX`, `TARGET_MEMORY_MAX_GB` for the architecture search constraints.
    *   If using RL-based search, adjust `RL_*` parameters.

2.  **Choose Search Algorithm in `main.py`**:
    *   By default, `main.py` might be set up to use one of the search algorithms.
    *   To use MIP-based search, ensure the import and call to `solve_mip` from `architecture_search.py` are active.
    *   To use RL-based search, ensure the import and call to `run_rl_search` from `rl_architecture_search.py` are active, and comment out the MIP part.

3.  **Run the main script**:
    ```bash
    python main.py
    ```

The script will proceed through the three stages: BLD & Scoring, Architecture Search, and GKD. Output and progress will be printed to the console.

### Changing the Parent Model

To use a different parent LLM:

1.  **Update `config.py`**:
    *   Change the `PARENT_MODEL_NAME` variable to the desired Hugging Face model identifier (e.g., `"EleutherAI/gpt-neo-125M"`, `"facebook/opt-125m"`).
    ```python
    # config.py
    PARENT_MODEL_NAME = "new/model-name-on-huggingface"
    ```

2.  **Adapt Utility Functions (`utils.py`)**:
    The functions `get_model_layers`, `set_model_layers`, `get_attention_block`, `get_ffn_block`, `set_attention_block`, and `set_ffn_block` are model-architecture-specific.
    *   You will likely need to add new `elif model.config.model_type == "new_model_type":` blocks to these functions to correctly access and modify the layers and sub-blocks of the new model. Refer to the Hugging Face Transformers documentation for the specific architecture of the new model.
    *   For example, if the new model has a different way to access its main transformer layers or its attention/FFN sub-modules, these functions must be updated.
    *   The `OPTFFNWrapper` was introduced for OPT models because their FFN is not a single module. Similar wrappers or logic might be needed for other unique architectures.

3.  **Adapt Block Variant Creation (`block_variants.py`)**:
    *   The `create_ffn_variant` function needs to know how to create reduced-dimension versions of the FFN block for the new model. This involves understanding the structure of the FFN (e.g., names of linear layers, how intermediate dimensions are defined).
    *   The `create_attention_variant` function (if implementing deep GQA or other complex attention modifications) would also need model-specific logic.

4.  **Adapt `local_distillation.py` (`train_single_block_variant`)**:
    *   **Embedding Layer Access**: The way initial embeddings are fetched might differ. Update the conditional logic based on `model_type`.
    *   **LayerNorm Access**: The names of Layer Normalization modules before FFN blocks can vary (e.g., `ln_2` for GPT-2/Neo, `final_layer_norm` for OPT). Update the conditional logic.
    *   **Attention Kwargs**: Ensure the `attn_kwargs` dictionary correctly prepares parameters (like `layer_past` vs `past_key_value`) for the new model's attention mechanism when calling attention sub-modules directly.

**Note**: Adapting to a new model architecture, especially one significantly different from GPT-2/Neo/OPT, can be a non-trivial task requiring careful study of its implementation in the Transformers library.

## Development Notes & Simplifications

*   **Hardware Cost Simulation**: This implementation uses highly simplified cost simulation (`simulate_block_cost` in `utils.py`). The original PUZZLE paper relies on precise measurements on target hardware (e.g., NVIDIA H100 with TensorRT-LLM).
*   **Dataset**: A small placeholder dataset (like "wikitext") is used for demonstration. Real-world application would require much larger and more diverse datasets for distillation.
*   **Scale**: The training epochs, number of samples, and token counts are significantly reduced for faster execution in a conceptual setting.
*   **GQA/Advanced Variants**: Deep implementation of Grouped-Query Attention (GQA) or other complex block variants is mostly placeholder.
*   **Error Handling**: Minimal error handling is implemented.

## References

This work is a conceptual re-implementation inspired by the following research:

*   **PUZZLE**: Bercovich, A., Ronen, T., Abramovich, T., et al. (2024). *PUZZLE: DISTILLATION-BASED NAS FOR INFERENCE-OPTIMIZED LLMS*. arXiv preprint arXiv:2411.19146. Available at: [https://www.arxiv.org/abs/2411.19146](https://www.arxiv.org/abs/2411.19146)
*   **LANA**: Molchanov, P., Hall, J., Yin, H., Kautz, J., Fusi, N., & Vahdat, A. (2021). *LANA: Latency Aware Network Acceleration*. arXiv preprint arXiv:2107.10624. Available at: [https://arxiv.org/abs/2107.10624](https://arxiv.org/abs/2107.10624)
*   **Nemotron-49B Example Model**: The Nemotron model series, such as [Llama-3.3-Nemotron-Super-49B-v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) by NVIDIA, showcases the kind of powerful models that can benefit from inference optimization techniques like PUZZLE.

## Contributing

This is a conceptual project. Contributions financeira improvements to the algorithmic flow, model adaptability, or more realistic simulations are welcome. Please open an issue or submit a pull request.
