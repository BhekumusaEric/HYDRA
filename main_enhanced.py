import os
import time
import logging
import argparse
import yaml
import ray
import subprocess
import docker
import matplotlib.pyplot as plt
import numpy as np
import threading
import requests
from tqdm import tqdm

# Import project modules
from red_agent.red_agent import train as train_red_agent
from blue_agent.blue_agent import train as train_blue_agent
from shared.replay_buffer import get_buffer
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

def setup_digital_twin():
    """
    Set up the digital twin environment using Docker Compose.
    """
    logger.info("Setting up digital twin environment...")
    
    try:
        # Check if Docker is running
        client = docker.from_env()
        client.ping()
        
        # Navigate to the twin_env directory and run docker-compose
        twin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_env")
        
        # Build and start the containers
        subprocess.run(
            ["docker-compose", "up", "-d", "--build"],
            cwd=twin_dir,
            check=True
        )
        
        logger.info("Digital twin environment is running")
        
        # Wait for services to be ready
        time.sleep(5)
        
        # Check if the web app is accessible
        try:
            import requests
            response = requests.get("http://localhost:5000/health")
            if response.status_code == 200:
                logger.info("Web application is ready")
            else:
                logger.warning(f"Web application returned status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to web application: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up digital twin: {e}")
        return False

def teardown_digital_twin():
    """
    Tear down the digital twin environment.
    """
    logger.info("Tearing down digital twin environment...")
    
    try:
        # Navigate to the twin_env directory and run docker-compose down
        twin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_env")
        
        # Stop the containers
        subprocess.run(
            ["docker-compose", "down"],
            cwd=twin_dir,
            check=True
        )
        
        logger.info("Digital twin environment has been stopped")
        return True
    except Exception as e:
        logger.error(f"Failed to tear down digital twin: {e}")
        return False

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

def run_graph_based_training(num_iterations=10, use_dqn=True, dashboard_enabled=True):
    """
    Run training using the graph-based environment and DQN agents.
    """
    logger.info(f"Starting graph-based training with {'DQN' if use_dqn else 'RLlib'} agents")
    
    # Create the graph environment
    env = GraphEnvironment(num_nodes=10, connectivity=0.3)
    
    # Get state and action dimensions
    state_size = len(env.get_state())
    red_action_size = env.num_nodes  # One action per node (attack)
    blue_action_size = env.num_nodes  # One action per node (defend)
    
    # Initialize agents
    if use_dqn:
        red_agent = RedDQNAgent(state_size, red_action_size)
        blue_agent = BlueDQNAgent(state_size, blue_action_size)
    else:
        # Use RLlib agents (not implemented here)
        logger.warning("RLlib agents not implemented for graph environment, falling back to DQN")
        red_agent = RedDQNAgent(state_size, red_action_size)
        blue_agent = BlueDQNAgent(state_size, blue_action_size)
    
    # Track rewards
    red_rewards = []
    blue_rewards = []
    
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

def run_co_training(num_iterations=10, red_config_path=None, blue_config_path=None, dashboard_enabled=True):
    """
    Run the co-training loop between Red and Blue agents using RLlib.
    """
    # Initialize Ray if not already initialized
    ray.init(ignore_reinit_error=True)
    
    # Get the shared replay buffer
    buffer = get_buffer()
    
    # Set default config paths if not provided
    if red_config_path is None:
        red_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "red_agent", "config.yaml")
    
    if blue_config_path is None:
        blue_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blue_agent", "config.yaml")
    
    # Load configurations
    red_config = load_config(red_config_path)
    blue_config = load_config(blue_config_path)
    
    # Track rewards for plotting
    red_rewards = []
    blue_rewards = []
    
    # Co-training loop
    for i in tqdm(range(num_iterations), desc="Co-Training Progress"):
        logger.info(f"Starting co-training iteration {i+1}/{num_iterations}")
        
        # Train Red Agent
        logger.info("Training Red Agent...")
        red_trainer, red_checkpoint = train_red_agent(red_config_path, num_iterations=1)
        red_result = red_trainer.train()
        red_rewards.append(red_result['episode_reward_mean'])
        logger.info(f"Red Agent training completed. Reward: {red_result['episode_reward_mean']}")
        
        # Train Blue Agent
        logger.info("Training Blue Agent...")
        blue_trainer, blue_checkpoint = train_blue_agent(blue_config_path, num_iterations=1)
        blue_result = blue_trainer.train()
        blue_rewards.append(blue_result['episode_reward_mean'])
        logger.info(f"Blue Agent training completed. Reward: {blue_result['episode_reward_mean']}")
        
        # Update dashboard
        if dashboard_enabled:
            update_dashboard(
                episode=i,
                red_reward=red_result['episode_reward_mean'],
                blue_reward=blue_result['episode_reward_mean'],
                network_state={
                    "compromised_nodes": np.random.randint(0, 5),  # Placeholder
                    "patched_nodes": np.random.randint(0, 5)       # Placeholder
                },
                log_message=f"Co-training iteration {i+1} completed"
            )
        
        # Save buffer state periodically
        if (i + 1) % 5 == 0 or i == num_iterations - 1:
            buffer.save()
        
        # Log buffer statistics
        buffer_stats = buffer.get_stats()
        logger.info(f"Buffer stats: {buffer_stats}")
    
    # Plot training results
    plot_training_results(red_rewards, blue_rewards)
    
    # Save final models
    final_red_checkpoint = red_trainer.save()
    final_blue_checkpoint = blue_trainer.save()
    
    logger.info(f"Co-training completed. Final models saved at:")
    logger.info(f"Red Agent: {final_red_checkpoint}")
    logger.info(f"Blue Agent: {final_blue_checkpoint}")
    
    return final_red_checkpoint, final_blue_checkpoint

def main():
    """
    Main entry point for the HYDRA system.
    """
    parser = argparse.ArgumentParser(description="Project HYDRA - Self-Evolving Red-Blue AI Symbiote")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--red-config", type=str, help="Path to Red Agent configuration")
    parser.add_argument("--blue-config", type=str, help="Path to Blue Agent configuration")
    parser.add_argument("--skip-twin", action="store_true", help="Skip digital twin setup/teardown")
    parser.add_argument("--use-dqn", action="store_true", help="Use DQN agents instead of RLlib")
    parser.add_argument("--use-graph", action="store_true", help="Use graph-based environment")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard")
    args = parser.parse_args()
    
    logger.info("Starting Project HYDRA")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Start dashboard if enabled
        dashboard_enabled = args.dashboard
        if dashboard_enabled:
            if not start_dashboard():
                logger.warning("Failed to start dashboard, continuing without it")
                dashboard_enabled = False
        
        # Set up digital twin environment
        if not args.skip_twin:
            if not setup_digital_twin():
                logger.error("Failed to set up digital twin. Exiting.")
                return
        
        # Run training
        if args.use_graph:
            # Run graph-based training
            run_graph_based_training(
                num_iterations=args.iterations,
                use_dqn=args.use_dqn,
                dashboard_enabled=dashboard_enabled
            )
        else:
            # Run traditional co-training
            run_co_training(
                num_iterations=args.iterations,
                red_config_path=args.red_config,
                blue_config_path=args.blue_config,
                dashboard_enabled=dashboard_enabled
            )
        
        # Tear down digital twin environment
        if not args.skip_twin:
            teardown_digital_twin()
        
        # Stop dashboard
        if dashboard_enabled:
            stop_dashboard()
        
        logger.info("Project HYDRA completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        if not args.skip_twin:
            teardown_digital_twin()
        if dashboard_enabled:
            stop_dashboard()
    
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        if not args.skip_twin:
            teardown_digital_twin()
        if dashboard_enabled:
            stop_dashboard()

if __name__ == "__main__":
    main()
