# Rainbow DQN Mode — 設計計畫書（最終版 v2）

> 第四個 mode：**☆ Rainbow DQN - Random Mode ☆**
> 目的：示範 Rainbow DQN 多技術整合（Double + Dueling + PER + Multi-step + NoisyNet + Distributional），並在**任意矩形網格 4 ≤ X ≤ 7, 4 ≤ Y ≤ 7** 下展現泛化能力。
> 流程：**離線預訓練** → 線上推論（同 Random / Compare 模式）。

---

## 1. 需求摘要

| 項目 | 規格 |
|---|---|
| 訓練流程 | 離線 `train_offline_rainbow.py` → `trained_rainbow.pth` → Flask 線上推論 |
| 網格尺寸 | X ∈ {4, 5, 6, 7}, Y ∈ {4, 5, 6, 7}（16 組） |
| 環境物件 | Goal × 1, Pit × 1, Wall × 1（**固定數量**） |
| 物件放置 | Player / Goal / Pit / Wall **每 episode 全部隨機 + 互不重疊** |
| UI 入口 | 上方下拉新增 `☆ Rainbow DQN - Random Mode ☆`（value=`rainbow_random`） |
| UI 樣式 | 仿 Lightning（隱藏 loss chart / sampled routes，顯示 info panel 與驗證 block） |
| Width / Height 控制位置 | 移到下方 Interactive Validation block 內（X, Y 各一個 4–7 select） |
| 驗證互動 | 可拖曳全部 4 個物件，重疊則 revert（仿 Lightning） |
| 探索 | NoisyNet 取代 ε-greedy（訓練 / 推論流程**無 ε 變數**） |

---

## 2. 模型架構

### 2.1 輸入維度策略 — Per-channel Zero Padding

`Gridworld.board.render_np()` 回傳 shape `(4, H, W)`，4 個 channel 依序是 **Player / Goal / Pit / Wall**（依 `components` dict 插入順序，見 [GridBoard.py:79–99](GridBoard.py)）。

**關鍵：必須在 3D tensor 階段做 padding，不能在 flatten 後 pad。**

```python
def pad_to_max(frame_3d, max_size=7):
    """frame_3d: (4, H, W) uint8/float → returns (4, 7, 7) float32"""
    assert frame_3d.shape[0] == 4, "expected 4 channels (Player/Goal/Pit/Wall)"
    C, H, W = frame_3d.shape
    out = np.zeros((C, max_size, max_size), dtype=np.float32)
    out[:, :H, :W] = frame_3d              # place at top-left; bottom/right zero-padded
    return out.flatten()                   # (4·7·7,) = (196,)
```

- 訓練 & 推論**共用此函式**（避免 train/eval skew）
- Padding 在右、下方向。每個 channel 的 49 維對齊
- 4 個 channel × 49 = **input_size = 196**

防呆：訓練腳本啟動時 `assert game.board.render_np().shape == (4, H, W)`，若未來 GridBoard 加 mask 會立即發現。

### 2.2 Rainbow 六大組件 — 已確認的決策

| 組件 | 實作要點 |
|---|---|
| **Double DQN** | action selection 用 online net，evaluation 用 target net |
| **Dueling** | V(s) 與 A(s, a) 各一條 head，Q = V + (A − A.mean(dim=action)) |
| **PER** | Proportional PER + SumTree（重用 `train_offline.SumTree` / `PrioritizedReplayBuffer`）；α=0.5, β: 0.4 → 1.0 |
| **Multi-step (n=3)** | NStepBuffer，γ^k 累積 reward；target 用 `s_{t+n}` 與 `γ^n` |
| **NoisyNet** | **Head-only**（最後 4 個 Linear 換成 NoisyLinear）；factorised Gaussian；**每次 forward 都 reset noise** |
| **C51** | 51 atoms, **V_min = -20, V_max = +10**（見下方推導）, categorical projection, cross-entropy loss |

#### V_min / V_max 推導
- 單 step reward ∈ {-1, -10, +10}
- n-step 累積 reward (n=3) worst case：`-1 + γ·(-1) + γ²·(-10) ≈ -11.7`
- 加上 `γ^n · z_i` bootstrapping (γ³ ≈ 0.97)：
  - target 下限 ≈ -11.7 + 0.97·(-10) ≈ -21.4 → V_min = **-20**（再寬一點更穩）
  - target 上限 ≈ -2 + 0.97·(+10) ≈ +7.7 → V_max = **+10**（向上對齊 reward）
- atom 間距 Δz = (V_max − V_min) / (N_atoms − 1) = 30 / 50 = 0.6

#### NoisyLinear 設計（Head-only 理由）
Rainbow 原論文是全 Linear 都換成 NoisyLinear，但 head-only 在 gridworld 這種小網路已足夠（noise 在低層會被高層稀釋）且訓練快 30%。我們選 **head-only**：

```
input (196)
  ↓ Linear(196 → 512) + ReLU        ← 共享 feature, 一般 Linear
  ↓ Linear(512 → 512) + ReLU        ← 共享 feature, 一般 Linear
  ↓ ─────────┬───────────────────────────────────┐
             ↓                                   ↓
       value branch                       advantage branch
       NoisyLinear(512 → 256) + ReLU     NoisyLinear(512 → 256) + ReLU
       NoisyLinear(256 → 51)             NoisyLinear(256 → 4·51)
             ↓                                   ↓
       V_logits: (B, 1, 51)             A_logits: (B, 4, 51)
             └──────────┬────────────────────────┘
                        ↓ broadcast 相加
       Q_logits = V + A − A.mean(dim=1, keepdim=True)   # (B, 4, 51)
                        ↓ softmax(dim=-1)
       Q_dist:   (B, 4, 51)              # 每個 action 的機率分佈
                        ↓ Σ_i  z_i · p_i
       Q(s, a):  (B, 4)                  # 用於 action selection
```

**NoisyLinear 細節：**
- factorised Gaussian noise（Rainbow 原論文版本）
- `forward()` 內每次自動 `reset_noise()`
- 有一個 `noisy: bool` flag，**驗證時設 False** 只用 μ（決定性，避免同 layout 兩次跑路徑不同）

### 2.3 C51 + n-step + PER 的 loss 偽碼

```python
# ────────  Target distribution (no grad)  ────────
with torch.no_grad():
    # Double DQN: argmax 用 online net，evaluate 用 target net
    a_star = online(s_next).argmax(dim=1)                 # (B,)
    p_next = softmax(target(s_next), dim=-1)              # (B, 4, 51)
    p_next_a = p_next[range(B), a_star]                   # (B, 51)

    Tz = r_n + (1 - done) * (gamma ** n) * z              # (B, 51), z=atom positions
    Tz = Tz.clamp(V_min, V_max)
    bj = (Tz - V_min) / delta_z                           # (B, 51) ∈ [0, 50]
    l = bj.floor().long()                                 # (B, 51)
    u = bj.ceil().long()                                  # (B, 51)
    # 防 l == u （bj 剛好整數）
    l[(u > 0) & (l == u)] -= 1
    u[(l < 50) & (l == u)] += 1

    m = torch.zeros(B, 51, device=device)
    m.scatter_add_(1, l, p_next_a * (u.float() - bj))
    m.scatter_add_(1, u, p_next_a * (bj - l.float()))

# ────────  Current distribution log-prob  ────────
log_p = log_softmax(online(s)[range(B), a], dim=-1)       # (B, 51)
ce_per_sample = -(m * log_p).sum(dim=1)                   # (B,)

# ────────  PER × C51  ────────
loss = (is_weights * ce_per_sample).mean()
td_errors = ce_per_sample.detach().cpu().numpy()          # 用 CE 當 priority
replay.update_priorities(idxs, td_errors)
```

---

## 3. 超參推薦

| 超參 | 值 | 備註 |
|---|---|---|
| EPOCHS（episodes） | **9,000**（curriculum，見 §3.1） | 1,500 + 2,000 + 2,500 + 3,000 |
| MAX_MOVES / episode | **`3 * (X + Y)`** 動態 | 4x4=24, 7x7=42 |
| BATCH_SIZE | **128** | C51 比 MSE 慢，batch 不需太大 |
| MEM_SIZE | **50,000** | 跨尺寸需多樣 |
| n-step | **3** | Rainbow 標準 |
| γ | **0.99** | 配 n-step 必要 |
| C51 N_atoms | **51** | 標準 |
| V_min / V_max | **-20 / +10** | 見 §2.2 推導 |
| Δz | **0.6** | (10 − (-20)) / 50 |
| lr (Adam) | **2.5e-4** → cosine → **1e-5** | NoisyNet 對 lr 敏感 |
| Target sync | 每 **1,000** env steps | 跨尺寸需穩定 |
| PER α / β | 0.5 / 0.4→1.0 線性退火 | Rainbow paper |
| NoisyNet σ₀ | **0.5** | 標準 |
| Grad clip (norm) | **10.0** | C51 cross-entropy 值較大 |
| Warm-up | **2,000** env steps（只填 buffer 不學習） | |
| ε-greedy | **不使用**（NoisyNet 取代） | |

### 3.1 訓練尺寸抽樣 — Curriculum

依 `max(X, Y)` 分四階段。**對稱性：** (X, Y) 與 (Y, X) 同難度，同階段一起學。

| Stage | max(X,Y) | 包含的 (X,Y) | episodes | 抽樣方式 |
|---|---|---|---|---|
| 1 | 4 | (4,4) | 1,500 | 純 4×4 |
| 2 | 5 | (4,5)(5,4)(5,5) | 2,000 | 80% 新尺寸 + 20% rehearsal stage 1 |
| 3 | 6 | (4,6)(6,4)(5,6)(6,5)(6,6) | 2,500 | 70% 新尺寸 + 30% rehearsal stage 1+2 |
| 4 | 7 | (4,7)(7,4)(5,7)(7,5)(6,7)(7,6)(7,7) | 3,000 | 60% 新尺寸 + 40% rehearsal 全部 |
| **合計** | | 16 組 | **9,000** | |

- **Replay buffer 跨 stage 不清空** — PER 自動 prioritize 新尺寸高 TD error 樣本
- Rehearsal 比例防 catastrophic forgetting

### 3.2 隨機 Layout 生成（自寫 helper，**不**用 `mode='random'`）

> 為何不用 `mode='random'`？ [Gridworld.initGridRand()](Gridworld.py:123) 內 pit/wall 數量由 `min(W,H)-1` 動態決定，4×4 是 1+1 但 7×7 可達 5+5。我們要的是**永遠 1+1+1+1**。

```python
def random_layout(W, H, max_attempts=50):
    """Return dict {Player, Goal, Pit, Wall} with non-overlapping positions
    AND BFS-reachable goal (player can reach goal avoiding wall, ignoring pit)."""
    for _ in range(max_attempts):
        positions = random.sample(
            [(r, c) for r in range(H) for c in range(W)], 4
        )
        layout = dict(zip(['Player', 'Goal', 'Pit', 'Wall'], positions))
        if _bfs_reachable(layout, W, H):   # player 可繞過 wall 抵達 goal
            return layout
    raise RuntimeError(f"Failed to generate solvable layout for {W}x{H}")

def _bfs_reachable(layout, W, H):
    """BFS from Player to Goal, treating Wall as blocker (Pit is walkable but lethal)."""
    # ...standard 4-neighbour BFS, return True if goal reachable
```

訓練時：
```python
layout = random_layout(W, H)
game = Gridworld(width=W, height=H, mode='player', custom_positions=layout)
```
（4 個 key 都傳，`Gridworld.__init__` 走 `elif custom_positions:` 分支直接吃。）

#### 跳過無解 layout 的理由
不做 BFS 也能跑，但 [validateBoard()](Gridworld.py:85) 只檢查角落基本情況，**沒檢查 wall 把 goal 圍住**。無解 layout 訓練時會 timeout，PER 給高 priority 反覆抽，網路會繞著不可解樣本打轉 → 必須做 BFS 預檢。

---

## 4. 預估訓練時間（2080 Ti + i7-10700）

**物件全 random 已包含在估算中。**

| Stage | 平均步數/ep | episodes | env steps |
|---|---|---|---|
| 1 (4×4) | ~12 | 1,500 | 18k |
| 2 (max 5) | ~16 | 2,000 | 32k |
| 3 (max 6) | ~22 | 2,500 | 55k |
| 4 (max 7) | ~28 | 3,000 | 84k |
| **合計** | | 9,000 | **≈ 190k env steps** |

| 場景 | 時間 |
|---|---|
| Stage 1 試水溫 | ~5 分鐘 |
| Stage 1+2 觀察曲線 | ~15 分鐘 |
| 完整 4-stage（推薦） | **~50–70 分鐘** |
| +2,000 ep stage 4 fine-tune | +20 分鐘 |

VRAM < 500 MB，網路 ~1.2M 參數（NoisyLinear 多一倍 σ）。瓶頸在 Python 環境迴圈，不在 GPU。

### 4.1 訓練監控指標

每 100 episodes 印一次：
- 每個 (X, Y) 的 success rate（last 100 episodes per size）
- PER β
- LR
- buffer size
- avg loss (cross-entropy)
- avg episode length

**不 log ε**（NoisyNet 不需要）。

---

## 5. UI 改動計畫

### 5.1 [templates/index.html](templates/index.html)

**A. 上方下拉新增（line 38–41）**
```html
<option value="rainbow_random">☆ Rainbow DQN - Random Mode ☆</option>
```

**B. 隱藏 / 顯示元素（Rainbow mode 下）**

| 元素 | 動作 |
|---|---|
| 頂部 `widthInput` / `heightInput` | 隱藏 |
| `epochs-group` / `train-btn` / `randomize-btn` | 隱藏（同 lightning/compare） |
| `lightning-controls` (grad-clip) | 隱藏 |
| `sampled-routes-panel` / `chart-panel-section` / `replay-panel-wrapper` | 隱藏 |
| `lightning-info-panel` / `lightning-image-panel` | 隱藏 |
| 新增 `#rainbow-info-panel` (Lightning 風格) | 顯示 |
| `#interactive-validation-panel` | 顯示，**改造**內部 |

**C. Interactive Validation Panel 改造（line 167 區塊）**

Rainbow mode 下：
- 隱藏 `num-walls` / `num-pits` input
- 新增 X / Y select（4–7）放在原 `num-walls/pits` 的位置
- 加一個 「⚡ Exploration: NoisyNet (replaces ε-greedy)」徽章

```html
<!-- Rainbow-only controls (hidden in lightning mode) -->
<div class="rainbow-only" style="display:none; gap:1.5rem;">
    <div class="control-group">
        <label>寬度 X:</label>
        <select id="rainbow-width-select">
            <option>4</option><option>5</option><option>6</option><option selected>7</option>
        </select>
    </div>
    <div class="control-group">
        <label>高度 Y:</label>
        <select id="rainbow-height-select">
            <option>4</option><option>5</option><option>6</option><option selected>7</option>
        </select>
    </div>
    <span class="badge" style="background:#7c3aed; color:#fff;">
        ⚡ NoisyNet (replaces ε-greedy)
    </span>
</div>
```

**D. `#rainbow-info-panel`** — Lightning 風格（文字 + 圖）

放在 lightning panel 下方，僅在 `rainbow_random` 顯示。內容：
- 標題：「Rainbow DQN — 六大技術整合」
- 左半：六組件文字（Double / Dueling / PER / Multi-step / NoisyNet / C51）+ Curriculum 4-stage 表
- 右半：`<img src="{{ url_for('static', filename='rainbow.png') }}">`

### 5.2 [static/app.js](static/app.js)

**A. mode 切換邏輯（line 451 起的 listener）**

```js
const isLightning = mode === 'lightning_random';
const isCompare   = mode === 'compare';
const isRainbow   = mode === 'rainbow_random';

// Rainbow 與 Lightning 共用的隱藏
const hideTrainingUI = isLightning || isCompare || isRainbow;

// Rainbow 獨有
rainbowInfoPanel.style.display = isRainbow ? 'block' : 'none';
document.querySelectorAll('.rainbow-only').forEach(el => {
    el.style.display = isRainbow ? 'flex' : 'none';
});
// 隱藏 lightning 的 num-walls/pits（Rainbow 不需要）
const lightningOnlyControls = [/* num-walls, num-pits */];
lightningOnlyControls.forEach(el => {
    el.style.display = isRainbow ? 'none' : '';
});

// Rainbow 共用 lightning 的 validation panel
interactiveValidationPanel.style.display = (isLightning || isRainbow) ? 'block' : 'none';

if (isRainbow) {
    const w = parseInt(document.getElementById('rainbow-width-select').value);
    const h = parseInt(document.getElementById('rainbow-height-select').value);
    initRainbowValidationGrid(w, h);
}
```

**B. `initRainbowValidationGrid(w, h)` — 全物件可拖曳**

- 隨機灑 4 個物件（呼叫前端版 `random_layout(w, h)`，不需要 BFS — 訓練模型已能應付不可解 layout 的近似最佳行為）
- 4 個 cell 全部 `draggable=true`
- dragstart：標記 source cell；cursor → grabbing；opacity 0.5
- drop：
  - 目標格為**空**：swap source.dataset.type ↔ target.dataset.type → 重繪 → 完成
  - 目標格**有其他物件**：revert（不交換），加 shake CSS 動畫提示
- X / Y select change → 重新隨機灑全部 4 個物件

```js
function shakeCell(cell) {
    cell.classList.add('shake');
    setTimeout(() => cell.classList.remove('shake'), 400);
}
```

`.shake` CSS 用 keyframe 0.4s 左右晃動。

**C. Verify 流程**

```js
async function verifyRainbow() {
    const w = parseInt(document.getElementById('rainbow-width-select').value);
    const h = parseInt(document.getElementById('rainbow-height-select').value);
    const positions = collectRainbowPositions();  // 從 grid 讀回 4 個座標
    const res = await fetch('/api/verify_rainbow', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ width: w, height: h, positions })
    });
    const data = await res.json();
    drawValidationRoute(data.path);  // 沿用現有 SVG 畫線
}
```

### 5.3 [app.py](app.py)

```python
RAINBOW_MAX_SIZE = 7
_rainbow_net = None   # lazy-loaded

def _load_rainbow():
    global _rainbow_net
    if _rainbow_net is None:
        from rl_models import RainbowDQN
        _rainbow_net = RainbowDQN(input_size=4 * RAINBOW_MAX_SIZE * RAINBOW_MAX_SIZE)
        _rainbow_net.load_state_dict(
            torch.load('trained_rainbow.pth', map_location='cpu')
        )
        _rainbow_net.eval()
        _rainbow_net.disable_noise()    # 推論時關閉 NoisyNet 噪聲
    return _rainbow_net

@app.route('/api/verify_rainbow', methods=['POST'])
def verify_rainbow():
    body = request.get_json()
    w, h = int(body['width']), int(body['height'])
    positions = body['positions']                  # dict of 4 entries
    return jsonify(verify_rainbow_model(w, h, positions))

def verify_rainbow_model(w, h, positions):
    net = _load_rainbow()
    layout = {k: tuple(v) for k, v in positions.items()}
    game = Gridworld(width=w, height=h, mode='player', custom_positions=layout)
    path = [tuple(game.board.components['Player'].pos)]
    for _ in range(3 * (w + h)):
        frame = game.board.render_np().astype(np.float32)
        s = pad_to_max(frame, RAINBOW_MAX_SIZE)            # (196,)
        s = torch.from_numpy(s).unsqueeze(0)
        with torch.no_grad():
            q = net(s)                                      # (1, 4) — already projected
            a = int(q.argmax(dim=1).item())
        game.makeMove({0:'u',1:'d',2:'l',3:'r'}[a])
        path.append(tuple(game.board.components['Player'].pos))
        r = game.reward()
        if abs(r) >= 10:
            break
    return {'path': path}
```

`pad_to_max` 從 `rl_models` import（同一份 code，避免 train/eval skew）。

### 5.4 [rl_models.py](rl_models.py)

新增類別：
- `NoisyLinear` — factorised Gaussian noise；`noisy: bool` flag；每次 forward auto reset noise
- `RainbowDQN` — 主網路；包含 `disable_noise()` / `enable_noise()` / `reset_noise()` 方法；`forward(s)` 預設回 Q-value（已經 `Σ z_i · p_i`）；另外提供 `forward_dist(s)` 回完整 distribution 用於訓練
- module-level `pad_to_max(frame_3d, max_size=7)` helper

### 5.5 拖曳重疊 UX

CSS（加進 `static/style.css` 或 inline）：
```css
@keyframes shake {
    0%,100% { transform: translateX(0); }
    25%     { transform: translateX(-4px); }
    75%     { transform: translateX(4px); }
}
.cell.shake { animation: shake 0.4s; }
```

---

## 6. 新檔案

| 檔案 | 用途 |
|---|---|
| `train_offline_rainbow.py` | 離線訓練主腳本 |
| `trained_rainbow.pth` | 訓練輸出（Git LFS 自動追蹤） |
| `static/rainbow.png` | UI info panel 配圖（架構或六組件示意圖） |

---

## 7. 現有檔案改動位置

| 檔案 | 改動 |
|---|---|
| `templates/index.html` | mode 下拉新增 `rainbow_random`；validation block 加 X/Y select 與 NoisyNet 徽章；新增 `#rainbow-info-panel` |
| `static/app.js` | `isRainbow` 分支；`initRainbowValidationGrid`；4 物件 drag-and-drop + shake；verify API |
| `static/style.css` | `.shake` keyframe |
| `app.py` | `/api/verify_rainbow`、`verify_rainbow_model`、`_load_rainbow`、`RAINBOW_MAX_SIZE` |
| `rl_models.py` | `NoisyLinear`、`RainbowDQN`、`pad_to_max` |
| `README.md` | 新增 Rainbow Mode 章節（架構、curriculum、訓練/驗證指令） |
| `.gitattributes` | 已涵蓋 `*.pth` LFS pattern，**不需改** |
| `requirements.txt` | 不需改（無新 dependency） |

---

## 8. 已確認決策一覽（含本輪追加）

| # | 項目 | 決策 |
|---|---|---|
| Q1 | NoisyNet vs ε-greedy | **NoisyNet**（head-only, factorised Gaussian, 每次 forward reset noise） |
| Q2 | 完整 Rainbow vs Rainbow-lite | **完整 Rainbow（含 C51）** |
| Q3 | 驗證互動性 | 4 物件全可拖曳；重疊則 revert + shake |
| Q4 | 訓練尺寸抽樣 | 4-stage curriculum（依 max(X,Y)），同階段含對稱組合 |
| Q5 | Info panel 風格 | Lightning 風格（圖 + 文字），`static/rainbow.png` |
| Q6 | UI 強調 NoisyNet | 紫色徽章「⚡ Exploration: NoisyNet (replaces ε-greedy)」 |
| R3 | C51 V_min / V_max | **-20 / +10**（n-step + bootstrap worst case 推導） |
| Y1 | NoisyLinear 套用範圍 | **Head-only**（前 2 層保留 Linear；省訓練時間且 gridworld 足夠） |
| Y2 | NoisyLinear reset_noise 時機 | **每次 forward** |
| Y4 | 無解 layout 處理 | 訓練端 **BFS 預檢**；驗證端不檢（讓 user 自由探索） |
| G2 | 拖曳重疊 UX | revert + CSS shake 動畫 |
| G3 | README 章節 | 訓練後補（不在計畫階段） |

---

## 9. HF Spaces 部署注意事項

- `torch.load(..., map_location='cpu')` 必加 — HF Spaces 是 CPU only
- 訓練完同樣需 `git push origin main && git push huggingface main`
- `trained_rainbow.pth` 自動被 `*.pth` LFS pattern 涵蓋，不需 `git lfs track` 手動加
- Dockerfile / requirements.txt 不需改動（純 PyTorch + Flask 已能跑 Rainbow）

---

## 10. 落地檢查清單

開新 session 動手時的順序：

- [ ] **Step 1**: `rl_models.py` 加 `NoisyLinear` + `RainbowDQN` + `pad_to_max`
  - [ ] 寫一個 `__main__` smoke test：dummy input (196,) → 確認 output shape (1, 4) 與 (1, 4, 51) 兩種 mode 都正常
  - [ ] `disable_noise()` / `enable_noise()` toggle test
- [ ] **Step 2**: `train_offline_rainbow.py`
  - [ ] 重用 `train_offline.SumTree` / `PrioritizedReplayBuffer`（直接 import 或拷貝）
  - [ ] 新增 `NStepBuffer`（暫存 n 步 transition，pop n-step return）
  - [ ] `random_layout(W, H)` + `_bfs_reachable()` helper
  - [ ] Curriculum scheduler：依 `(stage, episode_in_stage)` 決定 (X, Y) 抽樣分佈
  - [ ] training loop：env step → NStepBuffer → PER push → batch sample → C51 projection → CE loss × IS weight → priority update → 每 1000 step sync target
  - [ ] 監控 log：per-size success rate, β, LR, buffer size, loss
- [ ] **Step 3**: `app.py`：`/api/verify_rainbow` + `_load_rainbow()` + `verify_rainbow_model()`
- [ ] **Step 4**: `templates/index.html`：下拉選項、validation block 改造、`#rainbow-info-panel`
- [ ] **Step 5**: `static/app.js`：`isRainbow` 分支、`initRainbowValidationGrid` 全拖曳、verify call
- [ ] **Step 6**: `static/style.css`：`.shake` keyframe
- [ ] **Step 7**: 準備 `static/rainbow.png`（自製或網路找）
- [ ] **Step 8**: `python train_offline_rainbow.py` → 跑 stage 1 (~5 分鐘) 確認流程 → 跑完整 4 stage (~1 小時)
- [ ] **Step 9**: web 端手動驗證：每個 (X, Y) 至少跑 3 次拖曳測試
- [ ] **Step 10**: `README.md` 補 Rainbow 章節
- [ ] **Step 11**: `git push origin main && git push huggingface main`
