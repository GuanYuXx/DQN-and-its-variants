from flask import Flask, render_template, request, jsonify
from rl_models import train_dqn
import sys
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    try:
        data = request.json
        width = int(data.get('width', 4))
        height = int(data.get('height', 4))
        epochs = int(data.get('epochs', 1000))
        
        # Enforce minimum size
        if width < 4: width = 4
        if height < 4: height = 4
        
        # Run training
        results = train_dqn(width=width, height=height, epochs=epochs)
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
