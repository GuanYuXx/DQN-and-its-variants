"""
Offline Training Script for Rainbow DQN (Rainbow Random Mode)
────────────────────────────────────────────────────────────────────────
Rainbow = Double + Dueling + PER + Multi-step (n=3) + NoisyNet + C51

Training strategy:
- Single network, zero-padded inputs (4x7x7 = 196), works across any
  4 <= X <= 7, 4 <= Y <= 7 grid.
- Per-episode random layout (Goal/Pit/Wall/Player all randomised,
  non-overlapping, goal BFS-reachable from player ignoring pit).
- 4-stage curriculum on max(X, Y) with rehearsal to prevent forgetting.

Run:
    conda activate DQN
    python train_offline_rainbow.py

Output: trained_rainbow.pth (same directory)
"""

import random
from collections import deque

import numpy as np
import torch

from Gridworld import Gridworld
from rl_models import (
    RainbowDQN, pad_to_max,
    RAINBOW_MAX_SIZE, RAINBOW_N_ATOMS, RAINBOW_V_MIN, RAINBOW_V_MAX,
)
from train_offline import PrioritizedReplayBuffer  # reuse PER infra


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (see RAINBOW_DESIGN.md §3)
# ─────────────────────────────────────────────────────────────────────────────
GAMMA = 0.99
N_STEP = 3
BATCH_SIZE = 128
MEM_SIZE = 50_000
LR_START = 2.5e-4
LR_END = 1e-5
TARGET_SYNC_STEPS = 1_000
WARMUP_STEPS = 2_000
LEARN_EVERY = 4          # learn every k env steps (Rainbow paper default)
GRAD_CLIP_NORM = 10.0
# Set to 'cpu' for small networks like this one — Windows CUDA call overhead
# dominates the actual compute (1.2M params, 196-dim input is tiny).
# Override via env var:  set RAINBOW_DEVICE=cuda  python train_offline_rainbow.py
import os as _os
DEVICE_OVERRIDE = _os.environ.get('RAINBOW_DEVICE', 'cpu')
PER_ALPHA = 0.5
PER_BETA_START = 0.4
SAVE_PATH = 'trained_rainbow.pth'

# Curriculum: stage -> (episodes, new-size proportion, list of (X,Y) for "new")
CURRICULUM = [
    {'stage': 1, 'episodes': 1_500, 'new_pct': 1.00,
     'new_sizes': [(4, 4)],
     'rehearsal_sizes': []},
    {'stage': 2, 'episodes': 2_000, 'new_pct': 0.80,
     'new_sizes': [(4, 5), (5, 4), (5, 5)],
     'rehearsal_sizes': [(4, 4)]},
    {'stage': 3, 'episodes': 2_500, 'new_pct': 0.70,
     'new_sizes': [(4, 6), (6, 4), (5, 6), (6, 5), (6, 6)],
     'rehearsal_sizes': [(4, 4), (4, 5), (5, 4), (5, 5)]},
    {'stage': 4, 'episodes': 3_000, 'new_pct': 0.60,
     'new_sizes': [(4, 7), (7, 4), (5, 7), (7, 5), (6, 7), (7, 6), (7, 7)],
     'rehearsal_sizes': [(4, 4), (4, 5), (5, 4), (5, 5),
                         (4, 6), (6, 4), (5, 6), (6, 5), (6, 6)]},
]
TOTAL_EPISODES = sum(s['episodes'] for s in CURRICULUM)  # 9,000

# Estimate β annealing horizon from expected env steps.
# Rule of thumb: ~12-28 steps/episode depending on grid size; pick a value
# slightly larger than the expected total so β reaches ~1.0 near the end.
#   Stage 1 only (1,500 ep):  ~18k steps  → use 20_000
#   Full 4-stage  (9,000 ep): ~190k steps → use 200_000
ESTIMATED_TOTAL_STEPS = 20_000

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


# ─────────────────────────────────────────────────────────────────────────────
# Random layout helper  (do NOT use Gridworld.initGridRand — its pit/wall
# count grows with grid size; we always want exactly 1 of each.)
# ─────────────────────────────────────────────────────────────────────────────
def _bfs_reachable(layout, W, H):
    """BFS from Player to Goal, treating Wall as blocker (Pit is walkable)."""
    start = layout['Player']
    goal = layout['Goal']
    wall = layout['Wall']
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == wall:
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def random_layout(W, H, max_attempts=50):
    """Return dict {Player, Goal, Pit, Wall} with non-overlapping positions
    and BFS-reachable goal (player can reach goal avoiding wall, ignoring pit)."""
    cells = [(r, c) for r in range(H) for c in range(W)]
    for _ in range(max_attempts):
        positions = random.sample(cells, 4)
        layout = dict(zip(['Player', 'Goal', 'Pit', 'Wall'], positions))
        if _bfs_reachable(layout, W, H):
            return layout
    raise RuntimeError(f"Failed to generate solvable layout for {W}x{H}")


# ─────────────────────────────────────────────────────────────────────────────
# N-step buffer  (accumulates n transitions, emits n-step return)
# ─────────────────────────────────────────────────────────────────────────────
class NStepBuffer:
    """Holds up to n recent transitions for one episode and pops n-step rewards.

    Each call to .push() may yield ONE finalised n-step transition (or zero
    if the episode is too young). On episode end, call .flush() to drain
    remaining shorter transitions.
    """

    def __init__(self, n=N_STEP, gamma=GAMMA):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def __len__(self):
        return len(self.buffer)

    def _make_nstep(self, take):
        """Combine the first `take` transitions into one n-step transition."""
        s0, a0, _, _, _ = self.buffer[0]
        R = 0.0
        done_flag = 0.0
        s_next = self.buffer[take - 1][3]
        for i in range(take):
            _, _, r, s_n, d = self.buffer[i]
            R += (self.gamma ** i) * r
            if d:
                s_next = s_n
                done_flag = 1.0
                # truncate: ignore later rewards (episode already ended)
                return s0, a0, R, s_next, done_flag, i + 1
        return s0, a0, R, s_next, done_flag, take

    def push(self, transition):
        """transition = (state, action, reward, next_state, done).

        Returns the finalised n-step transition (s0, a0, R, s_n, done, n_actual)
        once the buffer is full, else None."""
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None
        out = self._make_nstep(self.n)
        # If the n-step window ended early on a `done`, drop the whole window;
        # otherwise drop just the oldest transition.
        if out[4] == 1.0 and out[5] < self.n:
            self.buffer.clear()
        else:
            self.buffer.popleft()
        return out

    def flush(self):
        """Drain any remaining transitions at end of episode (shorter n-step)."""
        outs = []
        while self.buffer:
            out = self._make_nstep(len(self.buffer))
            outs.append(out)
            self.buffer.popleft()
        return outs

    def reset(self):
        self.buffer.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum sampler
# ─────────────────────────────────────────────────────────────────────────────
def sample_size_for_stage(stage_cfg):
    """Pick (X, Y) for one episode based on the stage's new/rehearsal mix."""
    if not stage_cfg['rehearsal_sizes'] or random.random() < stage_cfg['new_pct']:
        return random.choice(stage_cfg['new_sizes'])
    return random.choice(stage_cfg['rehearsal_sizes'])


# ─────────────────────────────────────────────────────────────────────────────
# C51 categorical projection (vectorised, batch-wise)
# ─────────────────────────────────────────────────────────────────────────────
def project_distribution(next_dist, rewards, dones, gamma_n, support,
                         v_min, v_max, n_atoms):
    """Project Tz onto fixed support z. All inputs are torch tensors.

    Args:
        next_dist: (B, n_atoms)  p(z | s', a*) from target net under Double-DQN action
        rewards:   (B,)          n-step accumulated reward
        dones:     (B,)          0 or 1
        gamma_n:   (B,)          γ^k where k is the actual n-step length
        support:   (n_atoms,)    z atoms
        v_min, v_max, n_atoms: scalars

    Returns:
        m: (B, n_atoms)  projected target distribution
    """
    B = next_dist.size(0)
    device = next_dist.device
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # Tz = r + γ^k · z   (broadcast over atoms)
    Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * \
         gamma_n.unsqueeze(1) * support.unsqueeze(0)             # (B, n_atoms)
    Tz = Tz.clamp(min=v_min, max=v_max)
    bj = (Tz - v_min) / delta_z                                  # (B, n_atoms)
    l = bj.floor().long().clamp(min=0, max=n_atoms - 1)
    u = bj.ceil().long().clamp(min=0, max=n_atoms - 1)

    # Disappearing-mass fix for bj that lands exactly on an integer atom (l==u).
    # Apply *sequentially*: first try to shift l down by 1; for any remaining
    # l==u (only possible at the v_min boundary), shift u up by 1. This puts the
    # full mass on a single neighbour bin instead of splitting into two with
    # weight 1 each (which would double-count and violate Σp = 1).
    shift_l = (l == u) & (u > 0)
    l = torch.where(shift_l, l - 1, l)
    shift_u = (l == u) & (l < n_atoms - 1)
    u = torch.where(shift_u, u + 1, u)

    # Distribute probability mass
    m = torch.zeros(B, n_atoms, device=device)
    # lower bin gets weight (u - bj), upper bin gets weight (bj - l)
    m.scatter_add_(1, l, next_dist * (u.float() - bj))
    m.scatter_add_(1, u, next_dist * (bj - l.float()))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Episode rollout (one episode at one (W, H))
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(W, H, online_net, target_net, replay, nstep_buf,
                optimizer, device, global_step, success_log, loss_log):
    """Run one episode, push n-step transitions to PER, do gradient updates.

    Returns (env_steps_taken, success_bool, terminal_reward).
    """
    layout = random_layout(W, H)
    game = Gridworld(width=W, height=H, mode='player', custom_positions=layout)

    frame = game.board.render_np().astype(np.float32)
    assert frame.shape == (4, H, W), f"unexpected render shape {frame.shape}"
    state = pad_to_max(frame, RAINBOW_MAX_SIZE)  # (4, 7, 7) — CNN trunk
    state_t = torch.from_numpy(state).unsqueeze(0)
    if device.type != 'cpu':
        state_t = state_t.to(device)

    max_moves = 3 * (W + H)
    nstep_buf.reset()

    episode_done = False
    terminal_reward = -1
    env_steps = 0

    for _ in range(max_moves):
        # ── NoisyNet action selection ────────────────────────────────────
        # noise stays enabled during training
        with torch.no_grad():
            q = online_net(state_t)            # (1, 4)
            action = int(q.argmax(dim=1).item())

        game.makeMove(ACTION_SET[action])
        reward = game.reward()
        done = abs(reward) >= 10

        next_frame = game.board.render_np().astype(np.float32)
        next_state = pad_to_max(next_frame, RAINBOW_MAX_SIZE)

        # push to n-step buffer (only fully-formed n-step transitions are kept;
        # this guarantees γ^N_STEP is exact for the bootstrap term)
        nstep_out = nstep_buf.push((state, action, float(reward), next_state, float(done)))
        if nstep_out is not None:
            s0, a0, R, sN, dflag, _n_actual = nstep_out
            # store raw numpy — wrap to tensor only at batch time
            replay.push(s0, a0, R, sN, dflag)

        # gradient update (after warm-up, every LEARN_EVERY env steps)
        if (global_step[0] >= WARMUP_STEPS
                and len(replay) >= BATCH_SIZE
                and global_step[0] % LEARN_EVERY == 0):
            loss_val = _train_step(online_net, target_net, replay, optimizer, device)
            if loss_val is not None:
                loss_log.append(loss_val)

        # target sync
        if global_step[0] > 0 and global_step[0] % TARGET_SYNC_STEPS == 0:
            target_net.load_state_dict(online_net.state_dict())

        state = next_state
        state_t = torch.from_numpy(state).unsqueeze(0)
        if device.type != 'cpu':
            state_t = state_t.to(device)
        global_step[0] += 1
        env_steps += 1

        if done:
            terminal_reward = reward
            episode_done = True
            break

    # Drain remaining shorter n-step transitions — but ONLY if the episode
    # terminated. When max_moves expires without a terminal, the partial
    # n-step transitions would need γ^k != γ^N_STEP for an exact bootstrap,
    # which we don't track; safer to drop them than introduce a bias.
    if episode_done:
        for nstep_out in nstep_buf.flush():
            s0, a0, R, sN, dflag, _n_actual = nstep_out
            replay.push(s0, a0, R, sN, dflag)

    success = (terminal_reward == 10)
    success_log[(W, H)].append(1 if success else 0)
    return env_steps, success, terminal_reward


def _train_step(online_net, target_net, replay, optimizer, device):
    """One C51 + Double DQN + PER + n-step gradient step."""
    samples, idxs, is_weights = replay.sample(BATCH_SIZE)
    if not samples:
        return None

    # All replay entries have exactly n == N_STEP except those drained on
    # episode end with done=1, where (1-done)·γ^n=0 so γ^n is irrelevant.
    # Hence γ^N_STEP is exact for every gradient update.
    # Stack via numpy (much faster than torch.cat over many tiny tensors).
    states_np = np.stack([s[0] for s in samples])
    next_states_np = np.stack([s[3] for s in samples])
    states = torch.from_numpy(states_np).to(device)
    next_states = torch.from_numpy(next_states_np).to(device)
    actions = torch.tensor([s[1] for s in samples], dtype=torch.long, device=device)
    rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device)
    dones = torch.tensor([s[4] for s in samples], dtype=torch.float32, device=device)
    weights = torch.tensor(is_weights, dtype=torch.float32, device=device)

    # γ^n — use N_STEP uniformly (tail transitions have small fraction)
    gamma_n = torch.full((states.size(0),), GAMMA ** N_STEP,
                        dtype=torch.float32, device=device)

    n_atoms = online_net.n_atoms
    support = online_net.support
    v_min, v_max = online_net.v_min, online_net.v_max

    # ── Target distribution (Double DQN action selection) ─────────────────
    with torch.no_grad():
        # noise stays enabled — Rainbow paper uses NoisyNet noise also during target eval
        q_next_online = online_net(next_states)                      # (B, 4)
        a_star = q_next_online.argmax(dim=1)                         # (B,)

        p_next_all = target_net.forward_probs(next_states)           # (B, 4, n_atoms)
        idx = a_star.view(-1, 1, 1).expand(-1, 1, n_atoms)
        p_next = p_next_all.gather(1, idx).squeeze(1)                # (B, n_atoms)

        m = project_distribution(
            p_next, rewards, dones, gamma_n, support,
            v_min, v_max, n_atoms,
        )                                                            # (B, n_atoms)

    # ── Current log-probabilities for taken action ────────────────────────
    log_p_all = online_net.forward_dist(states)                      # (B, 4, n_atoms)
    idx = actions.view(-1, 1, 1).expand(-1, 1, n_atoms)
    log_p = log_p_all.gather(1, idx).squeeze(1)                      # (B, n_atoms)

    # Cross-entropy (per sample) — also used as PER priority
    ce = -(m * log_p).sum(dim=1)                                     # (B,)
    loss = (weights * ce).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    # priority update — use CE itself (positive scalar)
    td_errors = ce.detach().cpu().numpy()
    replay.update_priorities(idxs, td_errors)

    return float(loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Rainbow DQN  |  Double + Dueling + PER + N-step + Noisy + C51")
    print(f"  Curriculum:    {len(CURRICULUM)} stages, {TOTAL_EPISODES:,} episodes")
    print(f"  Grid range:    4..{RAINBOW_MAX_SIZE} (zero-padded to "
          f"{RAINBOW_MAX_SIZE}x{RAINBOW_MAX_SIZE})")
    print(f"  C51:           {RAINBOW_N_ATOMS} atoms, V∈[{RAINBOW_V_MIN}, {RAINBOW_V_MAX}]")
    print(f"  Exploration:   NoisyNet (no ε-greedy)")
    print("=" * 70)

    if DEVICE_OVERRIDE == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        # Use multiple cores for batched matmul on CPU
        torch.set_num_threads(max(1, (_os.cpu_count() or 4) // 2))
    print(f"Device: {device}  (override via RAINBOW_DEVICE env var)")

    online_net = RainbowDQN().to(device)
    target_net = RainbowDQN().to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR_START)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPISODES, eta_min=LR_END,
    )

    replay = PrioritizedReplayBuffer(
        capacity=MEM_SIZE,
        alpha=PER_ALPHA,
        beta_start=PER_BETA_START,
        beta_frames=ESTIMATED_TOTAL_STEPS,
    )

    nstep_buf = NStepBuffer(n=N_STEP, gamma=GAMMA)

    # rolling success log: { (W,H): deque(maxlen=100) }
    all_sizes = set()
    for stage in CURRICULUM:
        all_sizes.update(stage['new_sizes'])
        all_sizes.update(stage['rehearsal_sizes'])
    success_log = {sz: deque(maxlen=100) for sz in all_sizes}

    loss_log = deque(maxlen=500)
    global_step = [0]  # mutable holder so run_episode can update
    episode_global = 0

    for stage_cfg in CURRICULUM:
        stage = stage_cfg['stage']
        print(f"\n──── Stage {stage}  (max(X,Y) ≤ "
              f"{max(max(x, y) for x, y in stage_cfg['new_sizes'])})  "
              f"{stage_cfg['episodes']:,} episodes ────")

        for ep_in_stage in range(stage_cfg['episodes']):
            W, H = sample_size_for_stage(stage_cfg)
            run_episode(W, H, online_net, target_net, replay, nstep_buf,
                        optimizer, device, global_step, success_log, loss_log)
            scheduler.step()
            episode_global += 1

            if episode_global % 100 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                beta_now = replay.beta
                avg_loss = sum(loss_log) / max(1, len(loss_log))
                size_strs = []
                for sz in sorted(all_sizes):
                    log = success_log[sz]
                    if log:
                        sr = sum(log) / len(log)
                        size_strs.append(f"{sz[0]}x{sz[1]}:{sr:.0%}")
                sr_str = " ".join(size_strs)
                print(f"[ep {episode_global:>5d}/{TOTAL_EPISODES} step={global_step[0]:>6d}] "
                      f"loss={avg_loss:.3f}  lr={lr_now:.2e}  β={beta_now:.2f}  "
                      f"buf={len(replay):>5d}  | {sr_str}")

    # save weights
    torch.save(online_net.state_dict(), SAVE_PATH)
    print(f"\n✅  Model saved → {SAVE_PATH}")
    print(f"   {sum(p.numel() for p in online_net.parameters()):,} parameters")


if __name__ == '__main__':
    main()
