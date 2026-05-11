from flask import Flask, render_template, request, jsonify, Response
from rl_models import train_dqn_stream, train_comparison_stream, train_lightning_stream
import sys
import traceback

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
        return Response(train_comparison_stream(width=width, height=height, epochs=epochs, custom_positions=custom_positions if custom_positions else None), mimetype='text/event-stream')
    elif mode == 'lightning_random':
        return Response(train_lightning_stream(width=width, height=height, epochs=epochs, grad_clip=grad_clip), mimetype='text/event-stream')
    else:
        return Response(train_dqn_stream(width=width, height=height, epochs=epochs, custom_positions=custom_positions if custom_positions else None), mimetype='text/event-stream')

@app.route('/api/verify_random', methods=['POST'])
def verify_random():
    from rl_models import verify_lightning_model
    data = request.json
    width = data.get('width', 4)
    height = data.get('height', 4)
    custom_positions = data.get('custom_positions', {})
    
    # Convert lists to tuples; Pit/Wall may be arrays of positions [[r,c],[r,c],...]
    converted = {}
    for k, v in custom_positions.items():
        if k in ('Pit', 'Wall') and isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            converted[k] = [tuple(pos) for pos in v]
        else:
            converted[k] = tuple(v)
    custom_positions = converted

        
    try:
        route = verify_lightning_model(custom_positions, width=width, height=height)
        return jsonify({"status": "success", "route": route})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
