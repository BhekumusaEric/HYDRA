import os
import time
import logging
import argparse
import yaml
import ray
import subprocess
import docker
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project modules
from red_agent.red_agent import train as train_red_agent
from blue_agent.blue_agent import train as train_blue_agent
from shared.replay_buffer import get_buffer

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

def run_co_training(num_iterations=10, red_config_path=None, blue_config_path=None):
    """
    Run the co-training loop between Red and Blue agents.
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
    parser.add_argument("--iterations", type=int, default=10, help="Number of co-training iterations")
    parser.add_argument("--red-config", type=str, help="Path to Red Agent configuration")
    parser.add_argument("--blue-config", type=str, help="Path to Blue Agent configuration")
    parser.add_argument("--skip-twin", action="store_true", help="Skip digital twin setup/teardown")
    args = parser.parse_args()
    
    logger.info("Starting Project HYDRA")
    
    try:
        # Set up digital twin environment
        if not args.skip_twin:
            if not setup_digital_twin():
                logger.error("Failed to set up digital twin. Exiting.")
                return
        
        # Run co-training
        run_co_training(
            num_iterations=args.iterations,
            red_config_path=args.red_config,
            blue_config_path=args.blue_config
        )
        
        # Tear down digital twin environment
        if not args.skip_twin:
            teardown_digital_twin()
        
        logger.info("Project HYDRA completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        if not args.skip_twin:
            teardown_digital_twin()
    
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        if not args.skip_twin:
            teardown_digital_twin()

if __name__ == "__main__":
    main()
