# VIBID (Bidirectional Video Generation) 使用說明

## 功能簡介
VIBID 模式可以讓你同時指定開始和結束的影像，生成兩端一致的影片，實現完全雙向引導的影片生成。

## 使用方法

### 1. 準備起始和結尾影像
準備兩張圖片，分別作為影片的開始和結尾，例如：
```
assets/start_frame.jpg
assets/end_frame.jpg
```

### 2. 修改 YAML 配置文件
在你的配置文件中添加 VIBID 參數：

```yaml
- dreambooth_path: "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors"
  
  # VIBID 參數
  enable_vibid: True                           # 啟用 VIBID 模式
  start_image_path: "assets/start_frame.jpg"   # 起始影像路徑
  end_image_path: "assets/end_frame.jpg"       # 結尾影像路徑
  cfg_scale_flip: 1.0                          # 反向階段的 CFG 強度（可選，預設 1.0）
  n_inner: 5                                   # 資料一致性迭代次數（可選，預設 5）
  
  # 其他標準參數
  seed: [42]
  steps: 25
  guidance_scale: 8.0
  
  prompt:
    - "your prompt here"
  
  n_prompt:
    - "negative prompt here"
```

### 3. 執行生成
```bash
python scripts/animate.py \
    --config configs/prompts/1_animate/your_vibid_config.yaml \
    --L 16 --W 512 --H 512
```

## VIBID 參數說明

| 參數 | 必填 | 預設值 | 說明 |
|------|------|--------|------|
| `enable_vibid` | 是 | `False` | 啟用 VIBID 雙向生成模式 |
| `start_image_path` | 是* | - | 起始影像的檔案路徑 |
| `end_image_path` | 是* | - | 結尾影像的檔案路徑 |
| `cfg_scale_flip` | 否 | `1.0` | 反向階段的 CFG 引導強度 |
| `n_inner` | 否 | `5` | 共軛梯度法的內部迭代次數 |

*當 `enable_vibid=True` 時，`start_image_path` 和 `end_image_path` 都必須提供

## 參數調整建議

### cfg_scale_flip
- **較低值 (0.5-1.0)**：反向階段的引導較弱，生成更自然但可能不太符合 prompt
- **較高值 (1.0-2.0)**：反向階段的引導較強，更符合 prompt 但可能產生不自然的過渡

### n_inner
- **較低值 (3-5)**：速度較快，資料一致性較弱
- **較高值 (5-10)**：速度較慢，資料一致性較強，開始/結尾幀更接近提供的影像

## 範例配置

完整範例請參考：`configs/prompts/1_animate/vibid_example.yaml`

## 技術原理

VIBID 採用雙向去噪策略：

1. **正向階段 (Forward Pass)**：
   - 從起始影像編碼的 latent 開始去噪
   - 在每一步強制最後一幀接近結尾影像的 latent（透過 DDS 資料一致性）
   
2. **反向階段 (Backward Pass)**：
   - 將時間維度翻轉，從結尾往起始方向去噪
   - 強制反向的最後一幀（即原始的第一幀）接近起始影像的 latent
   - 翻轉回原始時間順序

這種雙向約束確保生成的影片在起始和結尾都嚴格符合提供的影像。

## 注意事項

1. **影像尺寸**：起始和結尾影像都會自動調整為配置中指定的尺寸 (W x H)
2. **VRAM 使用**：VIBID 模式會執行兩次前向傳播（正向+反向），VRAM 使用量約為標準模式的 1.5-2 倍
3. **生成時間**：由於需要雙向採樣，生成時間約為標準模式的 2 倍
4. **參考影像**：起始和結尾影像都會自動保存在輸出目錄的 `vibid_images/` 資料夾供參考

## 常見問題

### Q: 為什麼生成的影片開始/結尾與提供的影像不完全一致？
A: 可以嘗試增加 `n_inner` 參數值（例如從 5 增加到 10），這會增強資料一致性約束。

### Q: VIBID 可以與 ControlNet 一起使用嗎？
A: 目前的實作中，VIBID 主要專注於開始/結尾影像的一致性。如需同時使用，建議分開測試。

### Q: 如何停用 VIBID 回到標準模式？
A: 只需在 YAML 中設定 `enable_vibid: False` 或直接移除該參數即可。

### Q: 起始和結尾影像需要相似嗎？
A: 不需要。VIBID 的目標就是在兩個不同的影像之間生成平滑的過渡。但如果兩張影像差異過大，可能需要增加影片長度 (`--L`) 以獲得更自然的過渡效果。
