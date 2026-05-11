"""
Offline Training Script for Enhance DQN (Random Mode)
Uses Prioritized Experience Replay (Proportional PER) with IS weights,
Double DQN, Target Network, Gradient Clipping, and LR Scheduling.

Run this script independently:
    conda activate DQN
    python train_offline.py

Output: trained_lightning.pth (same directory)
"""

import torch
import torch.nn as nn
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from Gridworld import Gridworld
from GridBoard import BoardPiece  # noqa: F401  (imported for API compatibility)


# ─────────────────────────────────────────────────────────────────────────────
# SumTree  (O(log N) sampling & update)
# ─────────────────────────────────────────────────────────────────────────────
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def max_leaf_priority(self):
        """Return the max priority among actual leaf nodes (not internal nodes)."""
        leaf_start = self.capacity - 1
        if self.n_entries == 0:
            return 1.0
        leaf_end = leaf_start + self.n_entries
        return float(self.tree[leaf_start:leaf_end].max())


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Replay Buffer (PER)
# ─────────────────────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Proportional PER  →  p_i = (|δ_i| + ε)^α
    IS weight          →  w_i = (1 / N·P(i))^β  (normalised by max)
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-5          # prevent zero priority

    @property
    def beta(self):
        # anneal β from beta_start → 1.0
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        # FIX: use max_leaf_priority() which correctly looks at leaf nodes only
        max_p = self.tree.max_leaf_priority()
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None or (not isinstance(data, tuple)):
                continue
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        if len(batch) == 0:
            return [], [], []

        probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * probs, -self.beta)
        is_weights /= is_weights.max()
        return batch, idxs, is_weights.astype(np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, float(p))

    def __len__(self):
        return self.tree.n_entries


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, input_size, output_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module with PER
# ─────────────────────────────────────────────────────────────────────────────
class LitDQN_PER(pl.LightningModule):
    ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

    def __init__(self, width=4, height=4, max_moves=50,
                 gamma=0.9, lr=1e-3, batch_size=256,
                 mem_size=20_000, grad_clip=1.0,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.99997,
                 target_update_steps=500, max_epochs=5000):
        super().__init__()
        self.save_hyperparameters()
        self.width = width
        self.height = height
        self.input_size = 4 * width * height
        self.max_moves = max_moves
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.max_epochs = max_epochs

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.net = DQNNet(self.input_size)
        self.target_net = DQNNet(self.input_size)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.replay = PrioritizedReplayBuffer(capacity=mem_size)
        self.loss_fn = nn.MSELoss(reduction='none')

        self.automatic_optimization = False
        # FIX: track by step count so target syncs every N environment steps
        self._target_update_steps = target_update_steps
        self._global_step_count = 0

        self._game = None
        self._state = None
        self._moves = 0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _reset_game(self):
        self._game = Gridworld(width=self.width, height=self.height, mode='random')
        raw = self._game.board.render_np().astype(np.float32).flatten()
        self._state = torch.from_numpy(raw).unsqueeze(0)
        self._moves = 0

    def _state_on_device(self):
        return self._state.to(self.device)

    # ── Lightning hooks ───────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        # Cosine annealing: lr decays from lr → 1e-5 over all epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-5
        )
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_train_epoch_start(self):
        self._reset_game()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # ── Select action (ε-greedy) ──────────────────────────────────────
        s = self._state_on_device()
        with torch.no_grad():
            qvals = self.net(s)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = int(qvals.argmax(dim=1).item())

        # ── Step environment ──────────────────────────────────────────────
        self._game.makeMove(self.ACTION_SET[action_idx])
        reward = self._game.reward()

        # FIX: episode ends on BOTH goal (+10) and pit (-10), not just goal
        done = abs(reward) >= 10

        raw_next = self._game.board.render_np().astype(np.float32).flatten()
        next_state = torch.from_numpy(raw_next).unsqueeze(0)

        # ── Store transition ──────────────────────────────────────────────
        self.replay.push(self._state.cpu(), action_idx, reward, next_state.cpu(), float(done))
        self._state = next_state
        self._moves += 1
        self._global_step_count += 1

        # ── Sync target network every N steps ─────────────────────────────
        # FIX: was every 500 EPOCHS (=25k steps); now every 500 STEPS
        if self._global_step_count % self._target_update_steps == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # ── Learn (only when buffer is ready) ─────────────────────────────
        loss_val = 0.0
        if len(self.replay) >= self.batch_size:
            mini, idxs, is_w = self.replay.sample(self.batch_size)
            if mini:
                states   = torch.cat([m[0] for m in mini]).to(self.device)
                actions  = torch.tensor([m[1] for m in mini], dtype=torch.long, device=self.device)
                rewards  = torch.tensor([m[2] for m in mini], dtype=torch.float32, device=self.device)
                n_states = torch.cat([m[3] for m in mini]).to(self.device)
                dones    = torch.tensor([m[4] for m in mini], dtype=torch.float32, device=self.device)
                weights  = torch.tensor(is_w, dtype=torch.float32, device=self.device)

                # Double DQN target
                with torch.no_grad():
                    best_actions = self.net(n_states).argmax(dim=1, keepdim=True)
                    target_q = rewards + self.gamma * (1 - dones) * \
                               self.target_net(n_states).gather(1, best_actions).squeeze()

                curr_q = self.net(states).gather(1, actions.unsqueeze(1)).squeeze()
                element_loss = self.loss_fn(curr_q, target_q)

                td_errors = (target_q - curr_q).detach().cpu().numpy()
                self.replay.update_priorities(idxs, td_errors)

                loss = (weights * element_loss).mean()

                opt.zero_grad()
                self.manual_backward(loss)
                # FIX: clip_grad_norm_ (norm-based) is more stable than clip_grad_value_
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                opt.step()
                loss_val = loss.item()

        # ── Decay ε ───────────────────────────────────────────────────────
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ── Reset if episode ended ─────────────────────────────────────────
        if done or self._moves >= self.max_moves:
            self._reset_game()

        self.log('loss', loss_val, prog_bar=True)
        self.log('epsilon', self.epsilon, prog_bar=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dummy dataset  (steps_per_epoch samples per epoch)
# ─────────────────────────────────────────────────────────────────────────────
class DummyDataset(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, _): return torch.tensor([0.0])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    EPOCHS          = 5000   # 5000 epochs × 100 steps = 500k environment steps
    STEPS_PER_EPOCH = 100    # env steps per epoch
    WIDTH           = 4
    HEIGHT          = 4
    SAVE_PATH       = 'trained_lightning.pth'

    print("=" * 65)
    print("  Offline Training  |  PER + Double DQN + Target Net + LR Sched")
    print(f"  Grid: {WIDTH}x{HEIGHT}  |  Epochs: {EPOCHS:,}  |  Steps: {EPOCHS * STEPS_PER_EPOCH:,}")
    print("=" * 65)

    model = LitDQN_PER(
        width=WIDTH, height=HEIGHT,
        mem_size=50_000,
        batch_size=256,
        epsilon_decay=0.99997,  # reaches epsilon_min at ~100k steps (out of 500k)
        epsilon_min=0.05,
        max_moves=50,           # more moves allowed per episode
        target_update_steps=500,
        max_epochs=EPOCHS,
        grad_clip=1.0,
        lr=5e-4,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
    )

    trainer.fit(model, DataLoader(DummyDataset(STEPS_PER_EPOCH), batch_size=1))

    torch.save(model.net.state_dict(), SAVE_PATH)
    print(f"\n✅  Model saved → {SAVE_PATH}")
