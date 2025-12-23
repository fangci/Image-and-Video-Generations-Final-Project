# usage
- Sparse Control
    ```
    python -m scripts.animate --config configs/prompts/3_sparsectrl/3_3_sparsectrl_sketch_RealisticVision_forest.yaml
    python -m scripts.animate --config configs/prompts/3_sparsectrl/3_3_sparsectrl_sketch_RealisticVision_castle.yaml
    ```
- Background Masking
    ```
    python -m scripts.animate --config configs/prompts/1_animate/1_1_animate_RealisticVision_masl.yaml
    ```
- zero-shot layering
    ```
    python -m scripts.layer2 --config configs/prompts/1_animate/layer1.yaml
    python -m scripts.layer2 --config configs/prompts/1_animate/layer2.yaml
    python -m scripts.layer2 --config configs/prompts/1_animate/layer3.yaml
    ```