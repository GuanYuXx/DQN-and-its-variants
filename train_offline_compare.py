"""
Offline Training Script — Compare Mode (Double DQN vs Dueling DQN)
==================================================================

Trains BOTH agents offline against the same FIXED Goal/Pit/Wall layout
(initGridStatic positions), with the Player randomly placed each episode
(`mode='player'` in Gridworld). The resulting weights are loaded by the
Flask backend at verification time — the web UI no longer performs
on-line training for compare mode.

Run:
    conda activate DQN
    python train_offline_compare.py

Outputs (project root):
    trained_double.pth      Double DQN  state_dict
    trained_dueling.pth     Dueling DQN state_dict


─────────────────────────────────────────────────────────────────────
Fixed environment (matches Gridworld.initGridStatic for 4x4):
─────────────────────────────────────────────────────────────────────
    Goal  (+) at (0, 0)
    Pit   (-) at (0, 1)
    Wall  (W) at (1, 1)
    Player(P) randomized in remaining free cells per episode

Reward: +10 reach Goal, -10 fall into Pit, -1 per step otherwise.


─────────────────────────────────────────────────────────────────────
Architectures (imported from rl_models.py)
─────────────────────────────────────────────────────────────────────

Double DQN  (class DQNModel)
    State (4 * W * H = 64) → Linear(150) → ReLU
                           → Linear(100) → ReLU
                           → Linear(4)               # Q(s, a)
    Online + Target networks (same shape).
    Target rule:
        a*       = argmax_a  Q_online(s', a)
        Q_target = Q_target_net(s', a*)
        y        = r + γ · (1 − done) · Q_target
    → action-selection uses online, value-estimation uses target
      → mitigates the classic max-operator over-estimation bias.


Dueling DQN  (class DuelingDQNModel)
    shared feature : Linear(64) → ReLU → Linear(150)
    advantage head : Linear(150) → ReLU → Linear(100) → Linear(4)   # A(s, a)
    value     head : Linear(150) → ReLU → Linear(100) → Linear(1)   # V(s)
    Q(s, a) = V(s) + ( A(s, a) − mean_a A(s, a) )
    Target rule here: vanilla (max over target network) — keeps the
    architectural contribution isolated from the Double-target trick.


─────────────────────────────────────────────────────────────────────
Hyperparameters (shared)
─────────────────────────────────────────────────────────────────────
    epochs (episodes)        : 1500
    max moves / episode      : 50
    optimizer                : Adam, lr = 1e-3
    LR scheduler             : CosineAnnealingLR  (1e-3 → 1e-5)
    discount γ               : 0.9
    batch size               : 200
    replay buffer            : deque, capacity 10 000  (uniform sampling)
    target-net sync          : every 500 environment steps
    ε-greedy                 : 1.0 → 0.05 linearly over first 70% of epochs
    gradient clipping        : clip_grad_norm_ ≤ 1.0
    loss                     : MSE


─────────────────────────────────────────────────────────────────────
What's the *same*, what's *different* between the two runs?
─────────────────────────────────────────────────────────────────────
    SAME  : environment, hyperparameters, replay buffer mechanics,
            target network sync schedule, ε-schedule, optimizer, loss.
    DIFF  : (a) network architecture (single-head Q vs Value+Advantage)
            (b) target rule  (Double-target  vs  vanilla max-target)

This isolates the contributions of each technique fairly.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

from Gridworld import Gridworld
from rl_models import DQNModel, DuelingDQNModel


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
WIDTH              = 4
HEIGHT             = 4
EPOCHS             = 1500
MAX_MOVES          = 50
MEM_SIZE           = 10_000
BATCH_SIZE         = 200
GAMMA              = 0.9
LR                 = 1e-3
LR_MIN             = 1e-5
EPS_START          = 1.0
EPS_MIN            = 0.05
EPS_DECAY_FRACTION = 0.7         # ε reaches EPS_MIN at 70% of epochs
TARGET_SYNC_STEPS  = 500
GRAD_CLIP          = 1.0
EVAL_TRIALS        = 50

# Fixed environment (matches Gridworld.initGridStatic for 4x4)
CUSTOM_POSITIONS = {
    'Goal': (0, 0),
    'Pit':  (0, 1),
    'Wall': (1, 1),
}

INPUT_SIZE = 4 * WIDTH * HEIGHT
ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

SEED = 42


def _set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _state_tensor(game):
    """No noise — matches verification path in rl_models._verify_compare_rollout."""
    s = game.board.render_np().astype(np.float32).flatten()
    return torch.from_numpy(s).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Generic trainer — used for both Double DQN and Dueling DQN
# ─────────────────────────────────────────────────────────────────────────────
def train_agent(model_cls, double_target, label):
    """
    model_cls     : DQNModel or DuelingDQNModel
    double_target : True  → Double DQN target  (a* via online, eval via target)
                    False → vanilla target     (max over target net)
    label         : string used for logging
    """
    _set_seeds(SEED)

    online = model_cls(INPUT_SIZE)
    target = model_cls(INPUT_SIZE)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN
    )
    loss_fn = nn.MSELoss()
    replay  = deque(maxlen=MEM_SIZE)

    eps_decay_epochs = max(1, int(EPOCHS * EPS_DECAY_FRACTION))
    eps_step = (EPS_START - EPS_MIN) / eps_decay_epochs
    epsilon = EPS_START

    global_step = 0
    success_win = deque(maxlen=100)
    loss_win    = deque(maxlen=500)

    for ep in range(EPOCHS):
        game = Gridworld(width=WIDTH, height=HEIGHT,
                         mode='player', custom_positions=CUSTOM_POSITIONS)
        s = _state_tensor(game)
        moves = 0
        terminal_reward = -1

        while True:
            moves += 1

            # ── ε-greedy action selection ──
            with torch.no_grad():
                q = online(s)
            if random.random() < epsilon:
                a = random.randint(0, 3)
            else:
                a = int(q.argmax(dim=1).item())

            # ── env step ──
            game.makeMove(ACTION_SET[a])
            r = game.reward()
            done = abs(r) >= 10
            ns = _state_tensor(game)

            replay.append((s, a, r, ns, float(done)))
            s = ns
            global_step += 1

            # ── learn ──
            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)
                s_b  = torch.cat([m[0] for m in batch])
                a_b  = torch.tensor([m[1] for m in batch], dtype=torch.long)
                r_b  = torch.tensor([m[2] for m in batch], dtype=torch.float32)
                ns_b = torch.cat([m[3] for m in batch])
                d_b  = torch.tensor([m[4] for m in batch], dtype=torch.float32)

                with torch.no_grad():
                    if double_target:
                        next_a = online(ns_b).argmax(dim=1, keepdim=True)
                        q_next = target(ns_b).gather(1, next_a).squeeze(1)
                    else:
                        q_next = target(ns_b).max(dim=1)[0]
                    y = r_b + GAMMA * (1.0 - d_b) * q_next

                q_pred = online(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                loss = loss_fn(q_pred, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
                optimizer.step()
                loss_win.append(loss.item())

            # ── target-net sync ──
            if global_step % TARGET_SYNC_STEPS == 0:
                target.load_state_dict(online.state_dict())

            if done or moves >= MAX_MOVES:
                terminal_reward = r
                break

        success_win.append(1 if terminal_reward == 10 else 0)

        # ε decay (per epoch)
        if epsilon > EPS_MIN:
            epsilon = max(EPS_MIN, epsilon - eps_step)
        scheduler.step()

        if (ep + 1) % 100 == 0:
            sr   = sum(success_win) / max(1, len(success_win))
            avgL = sum(loss_win)    / max(1, len(loss_win))
            lrnow = scheduler.get_last_lr()[0]
            print(f"  [{label}] ep {ep+1:4d}/{EPOCHS}  "
                  f"success(last 100)={sr:6.1%}  loss={avgL:7.4f}  "
                  f"ε={epsilon:.3f}  lr={lrnow:.2e}")

    return online


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation: success rate from random Player starts
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(net, n_trials=EVAL_TRIALS):
    net.eval()
    successes, deaths, timeouts = 0, 0, 0
    avg_steps = 0.0
    for _ in range(n_trials):
        game = Gridworld(width=WIDTH, height=HEIGHT,
                         mode='player', custom_positions=CUSTOM_POSITIONS)
        s = _state_tensor(game)
        steps_taken = 0
        for _ in range(MAX_MOVES):
            with torch.no_grad():
                a = int(net(s).argmax(dim=1).item())
            game.makeMove(ACTION_SET[a])
            r = game.reward()
            steps_taken += 1
            if r == 10:
                successes += 1
                avg_steps += steps_taken
                break
            if r == -10:
                deaths += 1
                break
            s = _state_tensor(game)
        else:
            timeouts += 1
    avg_steps = (avg_steps / successes) if successes else 0.0
    return {
        'success_rate': successes / n_trials,
        'pit_rate':     deaths    / n_trials,
        'timeout_rate': timeouts  / n_trials,
        'avg_steps_when_success': avg_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 72)
    print("  Offline Compare-Mode Training | Double DQN  vs  Dueling DQN")
    print(f"  Grid {WIDTH}x{HEIGHT}   mode='player'   Player randomized per episode")
    print(f"  Goal={CUSTOM_POSITIONS['Goal']}  Pit={CUSTOM_POSITIONS['Pit']}  "
          f"Wall={CUSTOM_POSITIONS['Wall']}")
    print(f"  epochs={EPOCHS:,}  batch={BATCH_SIZE}  replay={MEM_SIZE:,}  γ={GAMMA}")
    print(f"  Adam lr {LR:.0e} → cosine → {LR_MIN:.0e}  |  target sync every "
          f"{TARGET_SYNC_STEPS} env steps")
    print("=" * 72)

    print("\n[1/2] Training Double DQN ...")
    double_net = train_agent(DQNModel, double_target=True, label='Double')
    torch.save(double_net.state_dict(), 'trained_double.pth')
    print("       saved → trained_double.pth")

    print("\n[2/2] Training Dueling DQN ...")
    dueling_net = train_agent(DuelingDQNModel, double_target=False, label='Dueling')
    torch.save(dueling_net.state_dict(), 'trained_dueling.pth')
    print("       saved → trained_dueling.pth")

    print("\n" + "=" * 72)
    print(f"  Greedy evaluation (random Player start, {EVAL_TRIALS} trials)")
    print("=" * 72)
    db = evaluate(double_net)
    du = evaluate(dueling_net)
    print(f"  Double  DQN : success {db['success_rate']:6.1%}  "
          f"pit {db['pit_rate']:5.1%}  timeout {db['timeout_rate']:5.1%}  "
          f"avg_steps={db['avg_steps_when_success']:.1f}")
    print(f"  Dueling DQN : success {du['success_rate']:6.1%}  "
          f"pit {du['pit_rate']:5.1%}  timeout {du['timeout_rate']:5.1%}  "
          f"avg_steps={du['avg_steps_when_success']:.1f}")
    print("=" * 72)
