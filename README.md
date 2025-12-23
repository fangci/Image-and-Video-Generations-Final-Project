最後有改動的檔案：
1. script: scripts/layer2.py
2. pipeline: animatediff/pipelines/pipeline_animation.py
3. attention: animatediff/models/attention.py
4. evaluation: eval.py

用來跑測試的config:
1. configs/prompts/1_animate/layer1.yaml
2. configs/prompts/1_animate/layer2.yaml
3. configs/prompts/1_animate/layer3.yaml

## usage
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