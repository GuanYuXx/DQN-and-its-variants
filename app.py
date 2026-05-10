from flask import Flask, render_template, request, jsonify, Response
from rl_models import train_dqn_stream
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
    
    return Response(train_dqn_stream(width=width, height=height, epochs=epochs, custom_positions=custom_positions if custom_positions else None), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
