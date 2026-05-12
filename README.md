---
title: DQN Variants Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# DQN and its Variants

互動式視覺化平台，在 Gridworld 環境中比較 Vanilla DQN、Double DQN、Dueling DQN、自行強化的 **Enhance DQN（PyTorch Lightning + PER）**，以及整合六項技術的 **Rainbow DQN（Double + Dueling + PER + Multi-step + NoisyNet + C51）** 的學習行為與泛化能力。

線上 Demo（Hugging Face Spaces）：<https://huggingface.co/spaces/GUanYuXx/DQN-variants-demo>

---

## 1. 開發環境（Development Environment）

| 項目 | 版本 / 設定 |
|------|------------|
| 作業系統 | Windows 11（Docker 部署為 `python:3.10-slim`） |
| Python | 3.10 |
| 深度學習框架 | PyTorch 2.5.1（CPU build）+ PyTorch Lightning 2.6.1 |
| Web 後端 | Flask 3.1.3（含 SSE 串流訓練進度） |
| 數值 / 視覺化 | NumPy 2.2.5、Matplotlib 3.10.9、torchmetrics 1.9.0 |
| 前端 | 原生 HTML / CSS / JavaScript + Chart.js（CDN）+ SVG 路徑繪圖 |
| 部署 | Hugging Face Spaces（Docker SDK），UID 1000 非 root 使用者，Port 7860 |
| 大型權重 | Git LFS 追蹤 `*.pth` 預訓練模型 |

完整套件清單請見 [`requirements.txt`](requirements.txt)。

### 啟動方式

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2.（可選）執行離線預訓練（會產生 .pth 權重）
python train_offline.py            # Enhance DQN → trained_lightning.pth
python train_offline_compare.py    # Double + Dueling DQN → trained_double.pth / trained_dueling.pth

# 3. 啟動互動式 Web 介面
python app.py
# 預設 http://localhost:7860
```

---

## 2. Gridworld 三種模式（Modes）

環境由 [`Gridworld.py`](Gridworld.py) 提供，預設為 4×4 棋盤。三種 `mode` 規範如下：

| 模式名稱<br>(`mode`) | 說明 | Player 位置 | 其他物件（Goal, Pit, Wall）位置 | 適用情境 |
|---|---|---|---|---|
| **`static`** | 完全靜態配置。所有物件位置固定 | 固定 **(0,3)** | 固定：<br>Goal → (0,0)<br>Pit → (0,1)<br>Wall → (1,1) | 用於測試邏輯正確性或可重現結果 |
| **`player`** | 只有 Player 隨機，其他物件固定 | 隨機位置 | 固定（同上） | 模擬不同起點，測試策略泛化能力 |
| **`random`** | 所有物件（Player, Goal, Pit, Wall）位置全隨機 | 隨機 | 隨機 | 用於訓練更穩健的策略，提升泛化能力 |

Web UI 各 mode 對應的功能：

| Web UI 選項 | 環境 mode | 模型 |
|---|---|---|
| Basic DQN (Original) | `static` | Vanilla DQN（線上訓練） |
| Compare: Double vs Dueling | `player` | 離線預訓練 Double / Dueling DQN |
| Enhance DQN (Random Mode – Lightning) | `random` | 離線預訓練 Enhance DQN |
| ☆ Rainbow DQN – Random Mode ☆ | `player` + 隨機尺寸 | 離線預訓練 Rainbow DQN（**任意 4–7 矩形**） |

---

## 3. 模型與技術

### 3.1 Double DQN（Compare 模式）

對應檔案：`DQNModel` in [`rl_models.py`](rl_models.py)

- **網路架構**：`Linear(64 → 150) → ReLU → Linear(150 → 100) → ReLU → Linear(100 → 4)`
- **核心想法**：將動作的「選擇」與「評估」解耦，緩解 Vanilla DQN 對 Q 值的過度估計。
- **Target rule（Double）**：
  ```
  y = r + γ · Q_target(s', argmax_a' Q_online(s', a'; θ); θ')
  ```
- 由 `train_offline_compare.py` 訓練，權重存於 `trained_double.pth`。

### 3.2 Dueling DQN（Compare 模式）

對應檔案：`DuelingDQNModel` in [`rl_models.py`](rl_models.py)

- **網路架構**：
  - Shared feature: `Linear(64 → 150) → ReLU`
  - Advantage head: `Linear(150 → 100) → ReLU → Linear(100 → 4)`
  - Value head：`Linear(150 → 100) → ReLU → Linear(100 → 1)`
  - 合併：`Q(s,a) = V(s) + (A(s,a) − mean_a' A(s,a'))`
- **核心想法**：分離「狀態價值 V(s)」與「動作優勢 A(s,a)」，當所有動作對結果差異不大時仍能學習狀態價值。
- **Target rule（Vanilla `max`）**：刻意採 vanilla target 來隔離「架構」對結果的貢獻（與 Double DQN 對比時公平）。
- 由 `train_offline_compare.py` 訓練，權重存於 `trained_dueling.pth`。

### 3.3 Enhance DQN（Random 模式，PyTorch Lightning + PER）

對應檔案：`LitDQN_PER` / `DQNNet` / `SumTree` / `PrioritizedReplayBuffer` in [`rl_models.py`](rl_models.py)，訓練腳本 [`train_offline.py`](train_offline.py)。

自行整合的「強化版」DQN，整合多種穩定技巧以對抗完全隨機環境：

| 技術 | 設定 / 重點 |
|---|---|
| **網路架構** | `Linear(input → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 4)` |
| **Prioritized Experience Replay (PER)** | Proportional PER + SumTree（O(log N) 抽樣 / 更新），α = 0.6，IS weight β 從 0.4 線性退火至 1.0，防零 ε = 1e-5 |
| **Double DQN target rule** | 與 3.1 相同，緩解 Q 值高估 |
| **Target Network 同步** | 每 **500 環境步**（step-based 而非 epoch-based）同步一次 |
| **Epsilon 衰減** | 指數衰減 `ε × 0.99997 / step`；1.0 → 0.05，約 100k 步進入近純利用階段 |
| **Learning Rate Scheduler** | `CosineAnnealingLR`：5×10⁻⁴ → 1×10⁻⁵，T_max = 5000 epochs |
| **Gradient Clipping** | `gradient_clip_val ≤ 1.0`，前端可即時調整 |
| **訓練規模** | 5000 epochs × 100 env steps = **50 萬環境步**；Buffer 容量 50,000；Batch Size 256 |
| **Lightning 封裝** | 訓練迴圈、Optimizer、Scheduler、硬體加速統一封裝於 `LitDQN_PER`，與 Web UI 完全解耦 |

### 3.4 Rainbow DQN（Random Mode，六合一強化版）

對應檔案：`RainbowDQN` / `NoisyLinear` / `pad_to_max` in [`rl_models.py`](rl_models.py)，訓練腳本 [`train_offline_rainbow.py`](train_offline_rainbow.py)，完整設計文件 [`RAINBOW_DESIGN.md`](RAINBOW_DESIGN.md)。

把 Rainbow paper（Hessel et al. 2017）的六項擴充全部整合在一個模型內，**且支援可變網格尺寸**（4×4 ~ 7×7 共 16 種矩形組合，包含非正方形如 4×6、7×5 等）。

#### (a) 六項技術組合

| # | 技術 | 在本實作中的角色 |
|---|---|---|
| 1 | **Double DQN** | argmax 用 online net、bootstrap value 用 target net，緩解過度估計 |
| 2 | **Dueling Network** | 共享 trunk 後分流 Value / Advantage head，per-atom 合併再 dueling-mean 校正 |
| 3 | **Prioritized Replay (PER)** | SumTree 實作，priority = per-sample C51 cross-entropy，α=0.6、β 線性退火 0.4→1.0 |
| 4 | **Multi-step Returns** | n=3 的 `NStepBuffer`，依 done flag 自動截斷，target 用 `γⁿ` |
| 5 | **NoisyNet** | Factorised Gaussian 取代 ε-greedy，每次 forward 自動 `reset_noise()`；**head-only**（前兩層 shared trunk 保留 deterministic Linear），推論時 `disable_noise()` |
| 6 | **Distributional RL (C51)** | 51 atoms，V_min=−10、V_max=+10，Δz=0.4；每個動作輸出 51 維機率分佈，Q(s,a)=Σ z_i·p_i |

#### (b) 網路架構（CNN trunk）

```
Input  (B, 4, 7, 7)   ← Player / Goal / Pit / Wall 四通道 + zero-pad
  │
Conv2d(4 → 32, 3×3, pad=1) → ReLU
Conv2d(32 → 64, 3×3, pad=1) → ReLU
  │
Flatten → Linear(3136 → 512) → ReLU       (shared trunk)
  ├──→ NoisyLinear(512 → 256) → ReLU → NoisyLinear(256 → 51)         (Value)
  └──→ NoisyLinear(512 → 256) → ReLU → NoisyLinear(256 → 4×51)       (Advantage)
       │
       └→ Dueling combine（per-atom）→ softmax → Q distribution (B, 4, 51)
```

選 CNN 而非 MLP 的關鍵：**空間平移不變性 + 參數共享**。4×4 與 7×7 的「Player 隔一格有 Pit」這個 pattern 在 MLP 看來是兩組完全無關的權重，CNN 直接共用同一個 3×3 kernel，是泛化到不同尺寸的根本機制。

#### (c) 可變尺寸：zero-pad to 7×7

`render_np()` 回傳 `(4, H, W)` 後用 `pad_to_max()` **per-channel** 補 0 到 `(4, 7, 7)`，**保持 3D 結構**讓 CNN 直接吃。padding 區固定為 0，CNN 自動學成「黑邊 = 不可走」。**注意必須先 pad 再 flatten，不能 flatten 後 pad**（會破壞 channel 對齊）。

#### (d) Curriculum Training（4 階段，9000 episodes）

| Stage | 包含 (X,Y) | Episodes | 新 / 舊回放比例 |
|---|---|---|---|
| 1 | (4,4) | 1500 | 100% / 0% |
| 2 | (4,5)(5,4)(5,5) | 2000 | 80% / 20% |
| 3 | + (4,6)(6,4)(5,6)(6,5)(6,6) | 2500 | 70% / 30% |
| 4 | + (4,7)(7,4)(5,7)(7,5)(6,7)(7,6)(7,7) | 3000 | 60% / 40% |

利用 (X,Y) 與 (Y,X) 對稱性同階段一起學；replay buffer 跨 stage 不清空，配合 PER 自動 prioritize 新尺寸的高 TD-error sample。

#### (e) 主要超參

| 項目 | 設定 |
|---|---|
| Discount γ | 0.99 |
| n-step | 3 |
| Batch size | 128（learn every 4 env steps） |
| Replay buffer | 50,000（PER：α=0.6, β 0.4→1.0） |
| Target sync | 每 1000 learn steps 硬同步 |
| Optimizer / LR | Adam, 2.5e-4 → 1e-5（cosine） |
| C51 atoms | 51, V_min=−10, V_max=+10, Δz=0.4 |
| NoisyLinear σ_init | 0.5 / √fan_in |
| Exploration | NoisyNet（**完全取代** ε-greedy） |

#### (f) 互動驗證

UI 內可選 Width / Height（皆 4–7），拖移 Player / Goal / Pit / Wall 任意位置（重疊會觸發 `.shake` 動畫回到原位）。按下 *Run Verification* → 呼叫 `/api/verify_rainbow` → 後端載入 `trained_rainbow.pth`、`disable_noise()` 後做 deterministic greedy rollout。

---

## 4. 離線模型（Pre-trained Models）

為了讓 Demo 在 Hugging Face Spaces 等 CPU-only 環境也能「秒開即用」，本專案採用「**離線訓練 → 線上推論**」的部署模式：

### 4.1 為什麼要離線訓練

- 在 Web 端做幾百 epoch 的線上訓練在 CPU 上往往不會收斂，且使用者等待時間長。
- 訓練邏輯（重）與部署推論（輕）解耦：[`app.py`](app.py) 僅在使用者按下 *Verify / Play* 時載入 `.pth` 做一次 greedy rollout。

### 4.2 預訓練檔案

| 檔案 | 來源腳本 | 對應 Web 模式 | 環境 mode | 主要超參 |
|---|---|---|---|---|
| `trained_lightning.pth` | `train_offline.py` | Enhance DQN (Random) | `random` | 5000 epochs，PER + Cosine LR + Target Net |
| `trained_double.pth` | `train_offline_compare.py` | Compare → Double 側 | `player` | 1500 epochs，Double target，Cosine LR |
| `trained_dueling.pth` | `train_offline_compare.py` | Compare → Dueling 側 | `player` | 1500 epochs，Vanilla target，Cosine LR |
| `trained_rainbow.pth` | `train_offline_rainbow.py` | ☆ Rainbow DQN (Random) | `player` + 隨機 4–7 尺寸 | 9000 episodes，4-stage curriculum，PER + n-step + NoisyNet + C51 |

四個檔案皆透過 **Git LFS** 追蹤（見 `.gitattributes` 的 `*.pth filter=lfs`），避免 git 倉庫膨脹。

### 4.3 互動驗證流程

- **Compare 模式（player）**
  1. UI 顯示 Goal/Pit/Wall 已固定於 (0,0)/(0,1)/(1,1)。
  2. 使用者點擊任一空白格設定 Player 起點。
  3. 按下 ▶ Play → 前端並行呼叫 `/api/verify_double` 與 `/api/verify_dueling`。
  4. 後端載入對應 `.pth`，於同一個地圖各做一次 greedy rollout，回傳路徑後在各自網格上繪製。

- **Enhance DQN 模式（random）**
  1. 使用者於 *Interactive Validation Grid* 任意拖移 Player / Goal / Pit / Wall（W、Pit 各至少 1 個，總數 ≤ min(X,Y) − 1）。
  2. 按下 *Run Verification* → 呼叫 `/api/verify_random`。
  3. 後端載入 `trained_lightning.pth` 在使用者擺出的地圖上做 greedy rollout，回傳路徑供前端繪製。

- **Rainbow DQN 模式（任意 4–7 矩形）**
  1. 使用者於 *Rainbow Interactive Validation Grid* 選擇 Width / Height（皆 4–7），並拖移 Player / Goal / Pit / Wall 至任意空格（重疊會 `.shake` 回原位）。
  2. 按下 *Run Verification* → 呼叫 `/api/verify_rainbow`。
  3. 後端載入 `trained_rainbow.pth`、呼叫 `disable_noise()` 後在 zero-pad 到 7×7 的輸入上做 deterministic greedy rollout，路徑以紫色 (#a78bfa) 繪製。

### 4.4 重新訓練

如果想自行調整超參數重新訓練，直接執行對應腳本即可（會覆寫專案根目錄的 `.pth` 檔）：

```bash
python train_offline.py             # 重新訓練 Enhance DQN
python train_offline_compare.py     # 重新訓練 Double + Dueling DQN
python train_offline_rainbow.py     # 重新訓練 Rainbow DQN（9000 ep curriculum）
```

---

## 專案結構

```
DQN-and-its-variants/
├── app.py                        # Flask 後端 + SSE 訓練串流 + verify API
├── Gridworld.py                  # 環境（static / player / random / custom 模式）
├── GridBoard.py                  # 棋盤元件
├── rl_models.py                  # DQNModel / DuelingDQNModel / LitDQN_PER / RainbowDQN / NoisyLinear
├── train_offline.py              # Enhance DQN 離線訓練
├── train_offline_compare.py      # Double + Dueling DQN 離線訓練
├── train_offline_rainbow.py      # Rainbow DQN 離線訓練（curriculum + C51 + PER + n-step）
├── RAINBOW_DESIGN.md             # Rainbow 完整設計文件
├── trained_lightning.pth         # ← LFS 追蹤
├── trained_double.pth            # ← LFS 追蹤
├── trained_dueling.pth           # ← LFS 追蹤
├── trained_rainbow.pth           # ← LFS 追蹤
├── templates/index.html          # 前端版面
├── static/
│   ├── app.js                    # 模式切換、拖移、SSE 接收、路徑繪製
│   ├── style.css
│   └── image.png                 # 架構圖
├── Dockerfile                    # HF Spaces 部署
└── requirements.txt
```
