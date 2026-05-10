# DQN and its Variants

This repository contains the implementation of a Deep Q-Network (DQN) with Experience Replay for a Gridworld environment.

## Features
- Dynamic Gridworld size (X * Y, where X, Y >= 4).
- DQN equipped with an Experience Replay Buffer for stable training.
- Interactive Flask-based web application.
- Real-time visualization of the training loss curve using Chart.js.
- Animated replay of the agent's actions on the grid upon successful training.

## Installation

We recommend using Miniconda to manage the environment.

1. Clone the repository:
   `ash
   git clone https://github.com/GuanYuXx/DQN-and-its-variants.git
   cd "DQN and its variants"
   `

2. Activate the conda environment (or create it if it doesn't exist):
   `ash
   conda create -n DQN python=3.10 flask numpy matplotlib pytorch torchvision torchaudio -c pytorch -y
   conda activate DQN
   `

## Usage

1. Start the Flask application:
   `ash
   python app.py
   `

2. Open your web browser and navigate to http://127.0.0.1:5000.

3. Configure the grid dimensions (Width and Height) and set the number of training Epochs.

4. Click **Start Training**. The backend will train the agent and, once complete, the loss chart will appear and the agent's gameplay will be replayed on the grid.

## Project Structure
- pp.py: Flask backend exposing the training API.
- l_models.py: PyTorch implementation of the DQN and the training loop.
- Gridworld.py / GridBoard.py: Gridworld environment logic.
- 	emplates/ & static/: Frontend interface and dynamic visualization logic.
