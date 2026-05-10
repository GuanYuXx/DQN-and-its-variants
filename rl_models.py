import numpy as np
import torch
import random
import json
from collections import deque
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
