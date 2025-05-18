import os
import time
import logging
import argparse
import yaml
import subprocess
import numpy as np
import threading
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import project modules
from deep_learning.dqn_agent import RedDQNAgent, BlueDQNAgent
from graph_env.graph_env import GraphEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hydra.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for dashboard communication
DASHBOARD_URL = "http://localhost:5001"
dashboard_process = None

def load_config(config_path):
    """
    Load configuration from YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}

def plot_training_results(red_rewards, blue_rewards, save_path="training_results.png"):
    """
    Plot training results for both agents.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Red Agent rewards
    plt.subplot(1, 2, 1)
    plt.plot(red_rewards, 'r-')
    plt.title('Red Agent Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot Blue Agent rewards
    plt.subplot(1, 2, 2)
    plt.plot(blue_rewards, 'b-')
    plt.title('Blue Agent Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training results plotted and saved to {save_path}")

def start_dashboard():
    """
    Start the dashboard in a separate process.
    """
    global dashboard_process
    
    logger.info("Starting HYDRA dashboard...")
    
    try:
        # Navigate to the dashboard directory and run the Flask app
        dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
        
        # Start the dashboard in a separate process
        dashboard_process = subprocess.Popen(
            ["python", "app.py"],
            cwd=dashboard_dir
        )
        
        # Wait for the dashboard to start
        time.sleep(3)
        
        # Check if the dashboard is accessible
        try:
            response = requests.get(f"{DASHBOARD_URL}/")
            if response.status_code == 200:
                logger.info("Dashboard is running at http://localhost:5001")
                return True
            else:
                logger.warning(f"Dashboard returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to dashboard: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return False

def stop_dashboard():
    """
    Stop the dashboard process.
    """
    global dashboard_process
    
    if dashboard_process:
        logger.info("Stopping HYDRA dashboard...")
        dashboard_process.terminate()
        dashboard_process = None

def update_dashboard(episode, red_reward, blue_reward, network_state, log_message=None, action=None):
    """
    Update the dashboard with the latest training information.
    """
    try:
        # Update training stats
        requests.post(
            f"{DASHBOARD_URL}/update_training",
            json={
                "episode": episode,
                "red_reward": red_reward,
                "blue_reward": blue_reward
            }
        )
        
        # Update network state
        requests.post(
            f"{DASHBOARD_URL}/update_network",
            json=network_state
        )
        
        # Add log if provided
        if log_message:
            requests.post(
                f"{DASHBOARD_URL}/add_log",
                json={
                    "message": log_message,
                    "level": "INFO",
                    "source": "main"
                }
            )
        
        # Add action if provided
        if action:
            requests.post(
                f"{DASHBOARD_URL}/add_action",
                json=action
            )
    
    except Exception as e:
        logger.warning(f"Failed to update dashboard: {e}")

def run_graph_based_training(num_iterations=10, dashboard_enabled=True):
    """
    Run training using the graph-based environment and DQN agents.
    """
    logger.info(f"Starting graph-based training with DQN agents")
    
    # Create the graph environment
    env = GraphEnvironment(num_nodes=10, connectivity=0.3)
    
    # Get state and action dimensions
    state_size = len(env.get_state())
    red_action_size = env.num_nodes  # One action per node (attack)
    blue_action_size = env.num_nodes  # One action per node (defend)
    
    # Initialize agents
    red_agent = RedDQNAgent(state_size, red_action_size)
    blue_agent = BlueDQNAgent(state_size, blue_action_size)
    
    # Track rewards
    red_rewards = []
    blue_rewards = []
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Training loop
    for episode in tqdm(range(num_iterations), desc="Training Progress"):
        # Reset environment
        state = env.reset()
        
        # Initialize episode variables
        done = False
        red_episode_reward = 0
        blue_episode_reward = 0
        step = 0
        
        # Generate a visualization of the initial network
        if episode % 5 == 0:
            env.visualize(f"network_episode_{episode}_initial.png")
        
        # Episode loop
        while not done and step < 50:  # Max 50 steps per episode
            # Red Agent's turn
            red_action = red_agent.act(state)
            success, reward, log_message = env.attack_node(red_action)
            red_episode_reward += reward
            
            # Log the action
            logger.info(f"Episode {episode}, Step {step}: Red Agent - {log_message}")
            
            # Update dashboard
            if dashboard_enabled:
                update_dashboard(
                    episode=episode,
                    red_reward=red_episode_reward,
                    blue_reward=blue_episode_reward,
                    network_state=env.get_network_stats(),
                    log_message=log_message,
                    action={
                        "type": "attack",
                        "agent": "red",
                        "details": {
                            "node_id": red_action,
                            "success": success,
                            "reward": reward
                        }
                    }
                )
            
            # Get new state
            next_state = env.get_state()
            
            # Store experience in Red Agent's memory
            red_agent.remember(state, red_action, reward, next_state, False)
            
            # Blue Agent's turn
            blue_action = blue_agent.act(next_state)
            success, reward, log_message = env.defend_node(blue_action)
            blue_episode_reward += reward
            
            # Log the action
            logger.info(f"Episode {episode}, Step {step}: Blue Agent - {log_message}")
            
            # Update dashboard
            if dashboard_enabled:
                update_dashboard(
                    episode=episode,
                    red_reward=red_episode_reward,
                    blue_reward=blue_episode_reward,
                    network_state=env.get_network_stats(),
                    log_message=log_message,
                    action={
                        "type": "defense",
                        "agent": "blue",
                        "details": {
                            "node_id": blue_action,
                            "success": success,
                            "reward": reward
                        }
                    }
                )
            
            # Get new state after Blue Agent's action
            next_state = env.get_state()
            
            # Store experience in Blue Agent's memory
            blue_agent.remember(state, blue_action, reward, next_state, False)
            
            # Update state
            state = next_state
            
            # Check if episode is done (all nodes compromised or all vulnerabilities patched)
            if len(env.compromised_nodes) == env.num_nodes or sum(1 for node in env.nodes.values() if node.patched) == env.num_nodes:
                done = True
            
            # Train agents
            if len(red_agent.memory) > red_agent.batch_size:
                red_loss = red_agent.replay()
                logger.debug(f"Red Agent loss: {red_loss}")
            
            if len(blue_agent.memory) > blue_agent.batch_size:
                blue_loss = blue_agent.replay()
                logger.debug(f"Blue Agent loss: {blue_loss}")
            
            step += 1
        
        # Generate a visualization of the final network
        if episode % 5 == 0:
            env.visualize(f"network_episode_{episode}_final.png")
        
        # Record rewards
        red_rewards.append(red_episode_reward)
        blue_rewards.append(blue_episode_reward)
        
        # Log episode results
        logger.info(f"Episode {episode} completed: Red Reward = {red_episode_reward}, Blue Reward = {blue_episode_reward}")
        
        # Save models periodically
        if episode % 10 == 0 or episode == num_iterations - 1:
            red_agent.save(f"models/red_dqn_episode_{episode}.pt")
            blue_agent.save(f"models/blue_dqn_episode_{episode}.pt")
    
    # Plot training results
    plot_training_results(red_rewards, blue_rewards)
    
    return red_agent, blue_agent, env

def main():
    """
    Main entry point for the HYDRA system.
    """
    parser = argparse.ArgumentParser(description="Project HYDRA - Self-Evolving Red-Blue AI Symbiote (Simplified Version)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard")
    args = parser.parse_args()
    
    logger.info("Starting Project HYDRA (Simplified Version)")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Start dashboard if enabled
        dashboard_enabled = args.dashboard
        if dashboard_enabled:
            if not start_dashboard():
                logger.warning("Failed to start dashboard, continuing without it")
                dashboard_enabled = False
        
        # Run graph-based training
        run_graph_based_training(
            num_iterations=args.iterations,
            dashboard_enabled=dashboard_enabled
        )
        
        # Stop dashboard
        if dashboard_enabled:
            stop_dashboard()
        
        logger.info("Project HYDRA completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        if dashboard_enabled:
            stop_dashboard()
    
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        if dashboard_enabled:
            stop_dashboard()

if __name__ == "__main__":
    main()
