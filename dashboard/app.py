from flask import Flask, render_template, jsonify, request
import os
import json
import logging
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Data storage (in a real app, this would be a database)
LOGS = []
NETWORK_STATE = {}
TRAINING_STATS = {
    "red_rewards": [],
    "blue_rewards": [],
    "episodes": []
}
ACTIONS_HISTORY = []

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/logs')
def get_logs():
    """Return the latest logs."""
    return jsonify(LOGS[-100:])  # Return the last 100 logs

@app.route('/network')
def get_network():
    """Return the current network state."""
    return jsonify(NETWORK_STATE)

@app.route('/training')
def get_training():
    """Return training statistics."""
    return jsonify(TRAINING_STATS)

@app.route('/actions')
def get_actions():
    """Return the history of actions taken."""
    return jsonify(ACTIONS_HISTORY[-100:])  # Return the last 100 actions

@app.route('/network_graph')
def get_network_graph():
    """Generate and return a visualization of the network graph."""
    try:
        # Create a simple graph for demonstration
        G = nx.erdos_renyi_graph(10, 0.3)
        
        # Assign node colors based on status (simulated)
        node_colors = []
        for node in G.nodes():
            # Simulate node status
            status = np.random.choice(['normal', 'compromised', 'patched'], p=[0.6, 0.2, 0.2])
            if status == 'compromised':
                node_colors.append('red')
            elif status == 'patched':
                node_colors.append('green')
            else:
                node_colors.append('blue')
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)  # For reproducibility
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_color='white')
        plt.title("Network Topology - Red: Compromised, Green: Patched, Blue: Normal")
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({'image': f'data:image/png;base64,{img_base64}'})
    
    except Exception as e:
        logger.error(f"Error generating network graph: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training_chart')
def get_training_chart():
    """Generate and return a visualization of training progress."""
    try:
        # Create sample data if empty
        if not TRAINING_STATS["episodes"]:
            TRAINING_STATS["episodes"] = list(range(1, 21))
            TRAINING_STATS["red_rewards"] = [np.random.normal(i/5, 1) for i in range(20)]
            TRAINING_STATS["blue_rewards"] = [np.random.normal(i/4, 1) for i in range(20)]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot Red Agent rewards
        plt.subplot(1, 2, 1)
        plt.plot(TRAINING_STATS["episodes"], TRAINING_STATS["red_rewards"], 'r-')
        plt.title('Red Agent Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot Blue Agent rewards
        plt.subplot(1, 2, 2)
        plt.plot(TRAINING_STATS["episodes"], TRAINING_STATS["blue_rewards"], 'b-')
        plt.title('Blue Agent Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({'image': f'data:image/png;base64,{img_base64}'})
    
    except Exception as e:
        logger.error(f"Error generating training chart: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_log', methods=['POST'])
def add_log():
    """Add a new log entry (API endpoint for the agents)."""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid log data'}), 400
        
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'message': data['message'],
            'level': data.get('level', 'INFO'),
            'source': data.get('source', 'unknown')
        }
        
        LOGS.append(log_entry)
        return jsonify({'status': 'success'})
    
    except Exception as e:
        logger.error(f"Error adding log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_network', methods=['POST'])
def update_network():
    """Update the network state (API endpoint for the agents)."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid network data'}), 400
        
        global NETWORK_STATE
        NETWORK_STATE = data
        return jsonify({'status': 'success'})
    
    except Exception as e:
        logger.error(f"Error updating network: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_training', methods=['POST'])
def update_training():
    """Update training statistics (API endpoint for the agents)."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid training data'}), 400
        
        if 'episode' in data and 'red_reward' in data and 'blue_reward' in data:
            TRAINING_STATS["episodes"].append(data['episode'])
            TRAINING_STATS["red_rewards"].append(data['red_reward'])
            TRAINING_STATS["blue_rewards"].append(data['blue_reward'])
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        logger.error(f"Error updating training stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_action', methods=['POST'])
def add_action():
    """Add a new action to the history (API endpoint for the agents)."""
    try:
        data = request.json
        if not data or 'type' not in data:
            return jsonify({'error': 'Invalid action data'}), 400
        
        action_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': data['type'],
            'agent': data.get('agent', 'unknown'),
            'details': data.get('details', {})
        }
        
        ACTIONS_HISTORY.append(action_entry)
        return jsonify({'status': 'success'})
    
    except Exception as e:
        logger.error(f"Error adding action: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
