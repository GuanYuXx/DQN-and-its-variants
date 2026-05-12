from flask import Flask, render_template, request, jsonify, Response
from rl_models import train_dqn_stream, train_comparison_stream, train_lightning_stream
import os
import sys
import traceback
import numpy as np
import torch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stream_train')
def stream_train():
    width = int(request.args.get('width', 4))
    height = int(request.args.get('height', 4))
    epochs = int(request.args.get('epochs', 500))
    
    if width < 4: width = 4
    if height < 4: height = 4
    
    custom_positions = {}
    for item in ['Player', 'Goal', 'Pit', 'Wall']:
        val = request.args.get(item.lower())
        if val:
            parts = val.split(',')
            if len(parts) == 2:
                custom_positions[item] = (int(parts[0]), int(parts[1]))
    
    mode = request.args.get('mode', 'basic')
    grad_clip = float(request.args.get('grad_clip', 1.0))
    
    if mode == 'compare':
        # Compare mode trains across random Player starts; never honour a Player hint from the client
        custom_positions.pop('Player', None)
        return Response(train_comparison_stream(width=width, height=height, epochs=epochs, custom_positions=custom_positions if custom_positions else None), mimetype='text/event-stream')
    elif mode == 'lightning_random':
        return Response(train_lightning_stream(width=width, height=height, epochs=epochs, grad_clip=grad_clip), mimetype='text/event-stream')
    else:
        return Response(train_dqn_stream(width=width, height=height, epochs=epochs, custom_positions=custom_positions if custom_positions else None), mimetype='text/event-stream')

def _convert_custom_positions(custom_positions):
    """Pit/Wall may be arrays of positions [[r,c],[r,c],...] or a single [r,c]."""
    converted = {}
    for k, v in custom_positions.items():
        if k in ('Pit', 'Wall') and isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            converted[k] = [tuple(pos) for pos in v]
        else:
            converted[k] = tuple(v)
    return converted


# Compare-mode offline models were trained against this fixed layout.
# See train_offline_compare.py:CUSTOM_POSITIONS — must match exactly.
COMPARE_FIXED_LAYOUT = {
    'Goal': (0, 0),
    'Pit':  (0, 1),
    'Wall': (1, 1),
}


@app.route('/api/verify_random', methods=['POST'])
def verify_random():
    from rl_models import verify_lightning_model
    data = request.json
    width = data.get('width', 4)
    height = data.get('height', 4)
    custom_positions = _convert_custom_positions(data.get('custom_positions', {}))

    try:
        route = verify_lightning_model(custom_positions, width=width, height=height)
        return jsonify({"status": "success", "route": route})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/verify_double', methods=['POST'])
def verify_double():
    from rl_models import verify_double_model
    data = request.json
    width = data.get('width', 4)
    height = data.get('height', 4)
    custom_positions = _convert_custom_positions(data.get('custom_positions', {})) or dict(COMPARE_FIXED_LAYOUT)
    player_pos = tuple(data['player_pos'])

    try:
        route = verify_double_model(custom_positions, player_pos, width=width, height=height)
        return jsonify({"status": "success", "route": route, "layout": COMPARE_FIXED_LAYOUT})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/verify_dueling', methods=['POST'])
def verify_dueling():
    from rl_models import verify_dueling_model
    data = request.json
    width = data.get('width', 4)
    height = data.get('height', 4)
    custom_positions = _convert_custom_positions(data.get('custom_positions', {})) or dict(COMPARE_FIXED_LAYOUT)
    player_pos = tuple(data['player_pos'])

    try:
        route = verify_dueling_model(custom_positions, player_pos, width=width, height=height)
        return jsonify({"status": "success", "route": route, "layout": COMPARE_FIXED_LAYOUT})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/compare_layout', methods=['GET'])
def compare_layout():
    """Return the fixed Goal/Pit/Wall layout the compare-mode models were trained against."""
    return jsonify({"layout": COMPARE_FIXED_LAYOUT})


# ─────────────────────────────────────────────────────────────────────────────
# Rainbow DQN inference (lazy-loaded; weights from train_offline_rainbow.py)
# ─────────────────────────────────────────────────────────────────────────────
RAINBOW_WEIGHTS_PATH = 'trained_rainbow.pth'
_rainbow_net = None


def _load_rainbow():
    """Lazy-load the Rainbow network. Throws FileNotFoundError if not trained yet."""
    global _rainbow_net
    if _rainbow_net is None:
        from rl_models import RainbowDQN
        if not os.path.exists(RAINBOW_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"{RAINBOW_WEIGHTS_PATH} not found — run `python train_offline_rainbow.py` first."
            )
        net = RainbowDQN()
        state_dict = torch.load(RAINBOW_WEIGHTS_PATH, map_location='cpu', weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        net.disable_noise()  # deterministic inference (no NoisyNet sampling)
        _rainbow_net = net
    return _rainbow_net


def verify_rainbow_model(width, height, positions):
    """Roll out greedy policy from the user-chosen layout. Returns dict with 'route'."""
    from rl_models import pad_to_max, RAINBOW_MAX_SIZE
    from Gridworld import Gridworld

    if not (4 <= width <= RAINBOW_MAX_SIZE and 4 <= height <= RAINBOW_MAX_SIZE):
        raise ValueError(f"width/height must be in [4, {RAINBOW_MAX_SIZE}]; got {width}x{height}")

    layout = {k: tuple(v) for k, v in positions.items()}
    required = {'Player', 'Goal', 'Pit', 'Wall'}
    missing = required - set(layout.keys())
    if missing:
        raise ValueError(f"missing positions: {missing}")

    net = _load_rainbow()
    game = Gridworld(width=width, height=height, mode='player', custom_positions=layout)

    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    player_pos = game.board.components['Player'].pos
    route = [{'pos': [int(player_pos[0]), int(player_pos[1])]}]

    max_moves = 3 * (width + height)
    for _ in range(max_moves):
        frame = game.board.render_np().astype(np.float32)
        s = pad_to_max(frame, RAINBOW_MAX_SIZE)
        s_t = torch.from_numpy(s).unsqueeze(0)
        with torch.no_grad():
            q = net(s_t)
            action = int(q.argmax(dim=1).item())

        game.makeMove(action_set[action])
        reward = game.reward()
        player_pos = game.board.components['Player'].pos
        route.append({
            'pos': [int(player_pos[0]), int(player_pos[1])],
            'reward': int(reward),
        })
        if abs(reward) >= 10:
            break

    return {'route': route}


@app.route('/api/verify_rainbow', methods=['POST'])
def verify_rainbow():
    data = request.get_json() or {}
    try:
        width = int(data.get('width', 7))
        height = int(data.get('height', 7))
        positions = data.get('positions', {})
        result = verify_rainbow_model(width, height, positions)
        return jsonify({'status': 'success', **result})
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'message': str(e)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
