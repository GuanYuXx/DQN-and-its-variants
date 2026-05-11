import numpy as np
import torch
import random
import json
import threading
import queue
import pytorch_lightning as pl
from collections import deque
from torch.utils.data import DataLoader, Dataset
from Gridworld import Gridworld

class DQNModel(torch.nn.Module):
    def __init__(self, input_size, l1=150, l2=100, l3=4):
        super(DQNModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, l1),
            torch.nn.ReLU(),
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3)
        )
    def forward(self, x):
        return self.model(x)

class DuelingDQNModel(torch.nn.Module):
    def __init__(self, input_size, l1=150, l2=100, l3=4):
        super(DuelingDQNModel, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(input_size, l1),
            torch.nn.ReLU()
        )
        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, 1)
        )
    def forward(self, x):
        f = self.feature(x)
        adv = self.advantage(f)
        val = self.value(f)
        return val + adv - adv.mean(dim=1, keepdim=True)

def train_dqn_stream(width=4, height=4, epochs=500, batch_size=200, mem_size=1000, max_moves=50, gamma=0.9, epsilon=1.0, epsilon_decay=True, custom_positions=None):
    input_size = 4 * width * height
    model = DQNModel(input_size)
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    replay = deque(maxlen=mem_size)
    
    interval = epochs // 5
    captured_routes = []
    final_game = []
    
    all_losses = []
    
    def running_mean(x, N=50):
        if len(x) == 0: return 0
        if len(x) < N: return sum(x) / len(x)
        c = len(x) - N
        y = np.zeros(c)
        conv = np.ones(N)
        for i in range(c):
            y[i] = (np.array(x[i:i+N]) @ conv)/N
        return y[-1]
    
    for i in range(epochs):
        game = Gridworld(width=width, height=height, mode='static', custom_positions=custom_positions)
        
        state1_ = game.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0
        state1 = torch.from_numpy(state1_).float()
        
        status = 1
        mov = 0
        current_route = []
        loss_count = 0
        
        while(status == 1):
            mov += 1
            
            player_pos = game.board.components['Player'].pos
            current_route.append({'pos': [int(player_pos[0]), int(player_pos[1])]})
            
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0
            state2 = torch.from_numpy(state2_).float()
            
            reward = game.reward()
            done = True if reward > 0 else False
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                loss_val = loss.item()
                all_losses.append(loss_val)
                loss_count += 1
                optimizer.step()
                
            if reward != -1 or mov > max_moves:
                status = 0
                player_pos = game.board.components['Player'].pos
                current_route.append({'pos': [int(player_pos[0]), int(player_pos[1])], 'reward': int(reward)})
                
        if epsilon_decay and epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if interval > 0 and (i + 1) % interval == 0:
            captured_routes.append({
                'epoch': i + 1,
                'route': current_route
            })
            
        if i == epochs - 1:
            final_game = current_route
            
        if loss_count > 0:
            smooth = running_mean(all_losses)
            # SSE uses "data: <content>\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'epoch': i + 1, 'loss': smooth})}\n\n"
            
    yield f"data: {json.dumps({'type': 'complete', 'routes': captured_routes, 'final_game': final_game})}\n\n"

def train_comparison_stream(width=4, height=4, epochs=500, batch_size=200, mem_size=1000, max_moves=50, gamma=0.9, epsilon=1.0, epsilon_decay=True, custom_positions=None):
    input_size = 4 * width * height
    
    # Double DQN Setup
    model_double = DQNModel(input_size)
    target_double = DQNModel(input_size)
    target_double.load_state_dict(model_double.state_dict())
    target_double.eval()
    optimizer_double = torch.optim.Adam(model_double.parameters(), lr=1e-3)
    replay_double = deque(maxlen=mem_size)
    
    # Dueling DQN Setup
    model_dueling = DuelingDQNModel(input_size)
    target_dueling = DuelingDQNModel(input_size)
    target_dueling.load_state_dict(model_dueling.state_dict())
    target_dueling.eval()
    optimizer_dueling = torch.optim.Adam(model_dueling.parameters(), lr=1e-3)
    replay_dueling = deque(maxlen=mem_size)
    
    loss_fn = torch.nn.MSELoss()
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    interval = epochs // 5
    captured_routes_double = []
    captured_routes_dueling = []
    final_game_double = []
    final_game_dueling = []
    
    all_losses_double = []
    all_losses_dueling = []
    
    def running_mean(x, N=50):
        if len(x) == 0: return 0
        if len(x) < N: return sum(x) / len(x)
        c = len(x) - N
        y = np.zeros(c)
        conv = np.ones(N)
        for i in range(c):
            y[i] = (np.array(x[i:i+N]) @ conv)/N
        return y[-1]
    
    sync_freq = 10  # Target network sync frequency
    
    for i in range(epochs):
        # ---------------- Double DQN Episode ----------------
        game_db = Gridworld(width=width, height=height, mode='player', custom_positions=custom_positions)
        state1_db_ = game_db.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0
        state1_db = torch.from_numpy(state1_db_).float()
        
        status_db = 1
        mov_db = 0
        current_route_db = []
        loss_count_db = 0
        
        while(status_db == 1):
            mov_db += 1
            player_pos = game_db.board.components['Player'].pos
            current_route_db.append({'pos': [int(player_pos[0]), int(player_pos[1])]})
            
            qval_db = model_double(state1_db)
            if (random.random() < epsilon):
                action_db_ = np.random.randint(0,4)
            else:
                action_db_ = np.argmax(qval_db.data.numpy())
            
            game_db.makeMove(action_set[action_db_])
            state2_db_ = game_db.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0
            state2_db = torch.from_numpy(state2_db_).float()
            reward_db = game_db.reward()
            done_db = True if reward_db > 0 else False
            
            replay_double.append((state1_db, action_db_, reward_db, state2_db, done_db))
            state1_db = state2_db
            
            if len(replay_double) > batch_size:
                minibatch = random.sample(replay_double, batch_size)
                s1_batch = torch.cat([s for (s,a,r,s2,d) in minibatch])
                a_batch = torch.Tensor([a for (s,a,r,s2,d) in minibatch])
                r_batch = torch.Tensor([r for (s,a,r,s2,d) in minibatch])
                s2_batch = torch.cat([s2 for (s,a,r,s2,d) in minibatch])
                d_batch = torch.Tensor([d for (s,a,r,s2,d) in minibatch])
                
                Q1 = model_double(s1_batch)
                with torch.no_grad():
                    Q_online_next = model_double(s2_batch)
                    argmax_a = torch.argmax(Q_online_next, dim=1)
                    Q_target_next = target_double(s2_batch)
                    Q2 = Q_target_next.gather(1, argmax_a.unsqueeze(1)).squeeze()
                
                Y = r_batch + gamma * ((1 - d_batch) * Q2)
                X = Q1.gather(dim=1, index=a_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                optimizer_double.zero_grad()
                loss.backward()
                all_losses_double.append(loss.item())
                loss_count_db += 1
                optimizer_double.step()
                
            if reward_db != -1 or mov_db > max_moves:
                status_db = 0
                player_pos = game_db.board.components['Player'].pos
                current_route_db.append({'pos': [int(player_pos[0]), int(player_pos[1])], 'reward': int(reward_db)})

        # ---------------- Dueling DQN Episode ----------------
        game_du = Gridworld(width=width, height=height, mode='player', custom_positions=custom_positions)
        state1_du_ = game_du.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0
        state1_du = torch.from_numpy(state1_du_).float()
        
        status_du = 1
        mov_du = 0
        current_route_du = []
        loss_count_du = 0
        
        while(status_du == 1):
            mov_du += 1
            player_pos = game_du.board.components['Player'].pos
            current_route_du.append({'pos': [int(player_pos[0]), int(player_pos[1])]})
            
            qval_du = model_dueling(state1_du)
            if (random.random() < epsilon):
                action_du_ = np.random.randint(0,4)
            else:
                action_du_ = np.argmax(qval_du.data.numpy())
            
            game_du.makeMove(action_set[action_du_])
            state2_du_ = game_du.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size) / 100.0
            state2_du = torch.from_numpy(state2_du_).float()
            reward_du = game_du.reward()
            done_du = True if reward_du > 0 else False
            
            replay_dueling.append((state1_du, action_du_, reward_du, state2_du, done_du))
            state1_du = state2_du
            
            if len(replay_dueling) > batch_size:
                minibatch = random.sample(replay_dueling, batch_size)
                s1_batch = torch.cat([s for (s,a,r,s2,d) in minibatch])
                a_batch = torch.Tensor([a for (s,a,r,s2,d) in minibatch])
                r_batch = torch.Tensor([r for (s,a,r,s2,d) in minibatch])
                s2_batch = torch.cat([s2 for (s,a,r,s2,d) in minibatch])
                d_batch = torch.Tensor([d for (s,a,r,s2,d) in minibatch])
                
                Q1 = model_dueling(s1_batch)
                with torch.no_grad():
                    Q2 = target_dueling(s2_batch)
                    
                Y = r_batch + gamma * ((1 - d_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=a_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                optimizer_dueling.zero_grad()
                loss.backward()
                all_losses_dueling.append(loss.item())
                loss_count_du += 1
                optimizer_dueling.step()
                
            if reward_du != -1 or mov_du > max_moves:
                status_du = 0
                player_pos = game_du.board.components['Player'].pos
                current_route_du.append({'pos': [int(player_pos[0]), int(player_pos[1])], 'reward': int(reward_du)})

        if epsilon_decay and epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % sync_freq == 0:
            target_double.load_state_dict(model_double.state_dict())
            target_dueling.load_state_dict(model_dueling.state_dict())
            
        if interval > 0 and (i + 1) % interval == 0:
            captured_routes_double.append({'epoch': i + 1, 'route': current_route_db})
            captured_routes_dueling.append({'epoch': i + 1, 'route': current_route_du})
            
        if i == epochs - 1:
            final_game_double = current_route_db
            final_game_dueling = current_route_du
            
        if loss_count_db > 0 or loss_count_du > 0:
            smooth_db = running_mean(all_losses_double)
            smooth_du = running_mean(all_losses_dueling)
            yield f"data: {json.dumps({'type': 'progress', 'epoch': i + 1, 'loss_double': smooth_db, 'loss_dueling': smooth_du})}\n\n"
            
    # Save trained models for later verification with user-chosen Player start
    torch.save(model_double.state_dict(), 'trained_double.pth')
    torch.save(model_dueling.state_dict(), 'trained_dueling.pth')

    yield f"data: {json.dumps({'type': 'complete', 'routes_double': captured_routes_double, 'final_game_double': final_game_double, 'routes_dueling': captured_routes_dueling, 'final_game_dueling': final_game_dueling})}\n\n"

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return 0

class LitDQN(pl.LightningModule):
    def __init__(self, input_size, width, height, max_moves=50, gamma=0.9, batch_size=200, mem_size=1000, grad_clip=1.0):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.height = height
        self.max_moves = max_moves
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.grad_clip = grad_clip
        
        self.avg_loss = 0.0
        self.current_lr = 0.0
        
        self.model = DQNModel(input_size)
        self.target = DQNModel(input_size)
        self.target.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.MSELoss()
        
        self.replay = deque(maxlen=mem_size)
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.automatic_optimization = False
        
        self.game = None
        self.current_route = []
        
    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.game = Gridworld(width=self.width, height=self.height, mode='random')
        self.state_np = self.game.board.render_np().reshape(1, self.input_size) + np.random.rand(1, self.input_size) / 100.0
        self.state_tensor = torch.from_numpy(self.state_np).float().to(self.device)
        self.status = 1
        self.mov = 0
        self.current_route = []

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        episode_loss = []
        
        while self.status == 1:
            self.mov += 1
            player_pos = self.game.board.components['Player'].pos
            self.current_route.append({'pos': [int(player_pos[0]), int(player_pos[1])]})
            
            qval = self.model(self.state_tensor)
            if random.random() < self.epsilon:
                action_idx = np.random.randint(0,4)
            else:
                action_idx = np.argmax(qval.detach().cpu().numpy())
                
            self.game.makeMove(self.action_set[action_idx])
            next_state_np = self.game.board.render_np().reshape(1, self.input_size) + np.random.rand(1, self.input_size) / 100.0
            next_state_tensor = torch.from_numpy(next_state_np).float().to(self.device)
            
            reward = self.game.reward()
            done = True if reward > 0 else False
            
            self.replay.append((self.state_tensor, action_idx, reward, next_state_tensor, done))
            self.state_tensor = next_state_tensor
            
            if len(self.replay) > self.batch_size:
                minibatch = random.sample(self.replay, self.batch_size)
                s1_batch = torch.cat([s for (s,a,r,s2,d) in minibatch]).to(self.device)
                a_batch = torch.Tensor([a for (s,a,r,s2,d) in minibatch]).to(self.device)
                r_batch = torch.Tensor([r for (s,a,r,s2,d) in minibatch]).to(self.device)
                s2_batch = torch.cat([s2 for (s,a,r,s2,d) in minibatch]).to(self.device)
                d_batch = torch.Tensor([d for (s,a,r,s2,d) in minibatch]).to(self.device)
                
                Q1 = self.model(s1_batch)
                with torch.no_grad():
                    Q2 = self.target(s2_batch)
                
                Y = r_batch + self.gamma * ((1 - d_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=a_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = self.loss_fn(X, Y.detach())
                
                opt.zero_grad()
                self.manual_backward(loss)
                if self.grad_clip > 0:
                    self.clip_gradients(opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
                opt.step()
                
                episode_loss.append(loss.item())
                
            if reward != -1 or self.mov > self.max_moves:
                self.status = 0
                player_pos = self.game.board.components['Player'].pos
                self.current_route.append({'pos': [int(player_pos[0]), int(player_pos[1])], 'reward': int(reward)})
        
        if self.epsilon > 0.1:
            self.epsilon -= (1/self.trainer.max_epochs)
            
        if (self.current_epoch + 1) % 10 == 0:
            self.target.load_state_dict(self.model.state_dict())
            
        if sch is not None:
            sch.step()
        
        avg_loss = sum(episode_loss)/len(episode_loss) if episode_loss else 0.0
        self.avg_loss = avg_loss
        self.current_lr = sch.get_last_lr()[0] if sch else opt.param_groups[0]['lr']
        self.log('train_loss', avg_loss)
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

class SSECallback(pl.Callback):
    def __init__(self, q, epochs):
        self.q = q
        self.epochs = epochs
        self.captured_routes = []
        self.final_game = []
        self.interval = epochs // 5
        self.all_losses = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        loss_val = getattr(pl_module, 'avg_loss', 0.0)
        lr_val = getattr(pl_module, 'current_lr', 0.0)
        self.all_losses.append(loss_val)
        
        smooth_loss = sum(self.all_losses[-10:])/len(self.all_losses[-10:]) if self.all_losses else 0.0
        epoch = trainer.current_epoch + 1
        
        self.q.put({"type": "progress", "epoch": epoch, "loss": smooth_loss, "lr": lr_val})
        
        route = list(pl_module.current_route)
        
        if self.interval > 0 and epoch % self.interval == 0:
            self.captured_routes.append({"epoch": epoch, "route": route})
            
        if epoch == self.epochs:
            self.final_game = route
            board = pl_module.game.board
            start_pos = tuple(route[0]['pos']) if route else board.components['Player'].pos
            init_pos = {
                "player": start_pos,
                "goal": board.components['Goal'].pos,
                "pit": board.components['Pit'].pos,
                "wall": board.components['Wall'].pos
            }
            self.q.put({"type": "complete", "routes": self.captured_routes, "final_game": self.final_game, "init_pos": init_pos})

def train_lightning_stream(width=4, height=4, epochs=500, batch_size=200, mem_size=1000, max_moves=50, gamma=0.9, grad_clip=1.0):
    input_size = 4 * width * height
    model = LitDQN(input_size, width, height, max_moves, gamma, batch_size, mem_size, grad_clip)
    q = queue.Queue()
    
    def run_trainer():
        dataset = DummyDataset(1)
        dataloader = DataLoader(dataset, batch_size=1)
        
        trainer = pl.Trainer(max_epochs=epochs, callbacks=[SSECallback(q, epochs)], enable_progress_bar=False, logger=False, accelerator='auto', devices=1)
        trainer.fit(model, train_dataloaders=dataloader)
        torch.save(model.state_dict(), 'trained_lightning.pth')
        q.put(None)
        
    threading.Thread(target=run_trainer).start()
    
    while True:
        msg = q.get()
        if msg is None:
            break
        yield f"data: {json.dumps(msg)}\n\n"

def verify_lightning_model(custom_positions, width=4, height=4):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from train_offline import DQNNet

    input_size = 4 * width * height
    net = DQNNet(input_size)
    state_dict = torch.load('trained_lightning.pth', map_location='cpu', weights_only=True)

    net.load_state_dict(state_dict)
    net.eval()

    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

    game = Gridworld(width=width, height=height, mode='static', custom_positions=custom_positions)
    state_np = game.board.render_np().astype(np.float32).flatten()
    state_tensor = torch.from_numpy(state_np).unsqueeze(0)

    route = [{'pos': [int(game.board.components['Player'].pos[0]),
                       int(game.board.components['Player'].pos[1])]}]
    max_moves = 20

    for _ in range(max_moves):
        with torch.no_grad():
            qval = net(state_tensor)
        action_idx = int(qval.argmax(dim=1).item())

        game.makeMove(action_set[action_idx])
        reward = game.reward()

        player_pos = game.board.components['Player'].pos
        route.append({'pos': [int(player_pos[0]), int(player_pos[1])], 'reward': int(reward)})

        if reward != -1:   # reached goal (+10) or fell into pit (-10)
            break

        next_np = game.board.render_np().astype(np.float32).flatten()
        state_tensor = torch.from_numpy(next_np).unsqueeze(0)

    return route


def _verify_compare_rollout(net, custom_positions, player_pos, width, height, max_moves=50):
    """Shared rollout: place Player at user-chosen start, run greedy policy until terminal or max_moves."""
    input_size = 4 * width * height
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

    cp = dict(custom_positions) if custom_positions else {}
    cp['Player'] = tuple(player_pos)

    game = Gridworld(width=width, height=height, mode='static', custom_positions=cp)

    state_np = game.board.render_np().astype(np.float32).flatten()
    state_tensor = torch.from_numpy(state_np).unsqueeze(0)

    route = [{'pos': [int(game.board.components['Player'].pos[0]),
                       int(game.board.components['Player'].pos[1])]}]

    for _ in range(max_moves):
        with torch.no_grad():
            qval = net(state_tensor)
        action_idx = int(qval.argmax(dim=1).item())

        game.makeMove(action_set[action_idx])
        reward = game.reward()

        player_pos_now = game.board.components['Player'].pos
        route.append({'pos': [int(player_pos_now[0]), int(player_pos_now[1])],
                      'reward': int(reward)})

        if reward != -1:  # terminal: +10 goal or -10 pit
            break

        next_np = game.board.render_np().astype(np.float32).flatten()
        state_tensor = torch.from_numpy(next_np).unsqueeze(0)

    return route


def verify_double_model(custom_positions, player_pos, width=4, height=4):
    input_size = 4 * width * height
    net = DQNModel(input_size)
    state_dict = torch.load('trained_double.pth', map_location='cpu', weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return _verify_compare_rollout(net, custom_positions, player_pos, width, height)


def verify_dueling_model(custom_positions, player_pos, width=4, height=4):
    input_size = 4 * width * height
    net = DuelingDQNModel(input_size)
    state_dict = torch.load('trained_dueling.pth', map_location='cpu', weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return _verify_compare_rollout(net, custom_positions, player_pos, width, height)
