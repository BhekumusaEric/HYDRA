import os
import yaml
import gym
import numpy as np
import requests
from gym import spaces
import logging
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HydraRedEnv(gym.Env):
    """
    Environment for the Red Agent in Project HYDRA.
    This environment simulates a target web application with vulnerabilities.
    """
    
    def __init__(self, config=None):
        config = config or {}
        
        # Load configuration
        self.target_host = config.get("target_host", "web-app")
        self.target_ports = config.get("target_ports", [5000, 5432])
        self.max_steps = config.get("max_steps_per_episode", 100)
        
        # Set up the action space
        # Actions: 
        # 0: Scan port 5000
        # 1: Scan port 5432
        # 2-11: Brute force login with different credentials
        # 12-15: SQL injection attempts
        # 16-19: Path traversal attempts
        self.action_space = spaces.Discrete(20)
        
        # Set up the observation space
        # Observations include:
        # - Port status (open/closed)
        # - Login attempt results
        # - Exploit success/failure
        # - Blue agent responses
        self.observation_space = spaces.Dict({
            "port_status": spaces.MultiBinary(len(self.target_ports)),
            "login_success": spaces.Discrete(2),
            "exploit_success": spaces.Discrete(2),
            "blocked": spaces.Discrete(2),
            "step_count": spaces.Discrete(self.max_steps + 1)
        })
        
        # Initialize state
        self.reset()
        
        logger.info("HydraRedEnv initialized")
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.port_status = np.zeros(len(self.target_ports), dtype=np.int8)
        self.login_success = 0
        self.exploit_success = 0
        self.blocked = 0
        
        # Initial observation
        obs = {
            "port_status": self.port_status,
            "login_success": self.login_success,
            "exploit_success": self.exploit_success,
            "blocked": self.blocked,
            "step_count": self.current_step
        }
        
        return obs
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): The action to take
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_step += 1
        reward = -0.01  # Small penalty for each step
        done = self.current_step >= self.max_steps
        info = {}
        
        # Process the action
        if action < len(self.target_ports):
            # Port scanning action
            port_idx = action
            port = self.target_ports[port_idx]
            success = self._scan_port(port)
            if success:
                self.port_status[port_idx] = 1
                reward += 0.1
                info["scan"] = f"Port {port} is open"
            else:
                info["scan"] = f"Port {port} is closed or scan was blocked"
                
        elif action < 12:
            # Brute force login action
            credentials = [
                ("admin", "admin"),
                ("admin", "password"),
                ("admin", "secret123"),
                ("user1", "user1"),
                ("user1", "password123"),
                ("user", "user"),
                ("guest", "guest"),
                ("root", "root"),
                ("admin", "123456"),
                ("test", "test")
            ]
            cred_idx = action - 2
            if cred_idx < len(credentials):
                username, password = credentials[cred_idx]
                success, blocked = self._attempt_login(username, password)
                if success:
                    self.login_success = 1
                    reward += 1.0
                    info["login"] = f"Successful login as {username}"
                elif blocked:
                    self.blocked = 1
                    reward -= 0.5
                    info["login"] = "Login attempt was blocked"
                else:
                    info["login"] = "Login failed"
        
        elif action < 16:
            # SQL injection action
            payloads = [
                "' OR '1'='1", 
                "'; DROP TABLE users; --", 
                "' UNION SELECT * FROM users; --",
                "admin' --"
            ]
            payload_idx = action - 12
            if payload_idx < len(payloads):
                payload = payloads[payload_idx]
                success, blocked = self._attempt_sql_injection(payload)
                if success:
                    self.exploit_success = 1
                    reward += 2.0
                    info["sql_injection"] = "SQL injection successful"
                elif blocked:
                    self.blocked = 1
                    reward -= 0.5
                    info["sql_injection"] = "SQL injection attempt was blocked"
                else:
                    info["sql_injection"] = "SQL injection failed"
        
        else:
            # Path traversal action
            paths = [
                "../../../etc/passwd",
                "../../../../etc/shadow",
                "../config/database.yml",
                "../../.env"
            ]
            path_idx = action - 16
            if path_idx < len(paths):
                path = paths[path_idx]
                success, blocked = self._attempt_path_traversal(path)
                if success:
                    self.exploit_success = 1
                    reward += 2.0
                    info["path_traversal"] = "Path traversal successful"
                elif blocked:
                    self.blocked = 1
                    reward -= 0.5
                    info["path_traversal"] = "Path traversal attempt was blocked"
                else:
                    info["path_traversal"] = "Path traversal failed"
        
        # Construct observation
        obs = {
            "port_status": self.port_status,
            "login_success": self.login_success,
            "exploit_success": self.exploit_success,
            "blocked": self.blocked,
            "step_count": self.current_step
        }
        
        return obs, reward, done, info
    
    def _scan_port(self, port):
        """Simulate port scanning."""
        try:
            # In a real implementation, this would actually check if the port is open
            # For simulation, we'll assume the ports in target_ports are open
            return port in self.target_ports
        except Exception as e:
            logger.error(f"Error scanning port {port}: {e}")
            return False
    
    def _attempt_login(self, username, password):
        """Simulate a login attempt."""
        try:
            # In a real implementation, this would make an actual HTTP request
            # For simulation, we'll check against our known credentials
            valid_credentials = {
                "admin": "secret123",
                "user1": "password123",
                "guest": "guest"
            }
            
            success = username in valid_credentials and valid_credentials[username] == password
            # Simulate Blue Agent blocking (random chance for now)
            blocked = np.random.random() < 0.2  # 20% chance of being blocked
            
            return success and not blocked, blocked
        except Exception as e:
            logger.error(f"Error attempting login: {e}")
            return False, False
    
    def _attempt_sql_injection(self, payload):
        """Simulate an SQL injection attempt."""
        try:
            # In a real implementation, this would make an actual HTTP request
            # For simulation, we'll check if the payload contains SQL injection patterns
            success = "'" in payload or ";" in payload
            # Simulate Blue Agent blocking (random chance for now)
            blocked = np.random.random() < 0.3  # 30% chance of being blocked
            
            return success and not blocked, blocked
        except Exception as e:
            logger.error(f"Error attempting SQL injection: {e}")
            return False, False
    
    def _attempt_path_traversal(self, path):
        """Simulate a path traversal attempt."""
        try:
            # In a real implementation, this would make an actual HTTP request
            # For simulation, we'll check if the path contains path traversal patterns
            success = "../" in path
            # Simulate Blue Agent blocking (random chance for now)
            blocked = np.random.random() < 0.3  # 30% chance of being blocked
            
            return success and not blocked, blocked
        except Exception as e:
            logger.error(f"Error attempting path traversal: {e}")
            return False, False

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def register_environments():
    """Register custom environments with Ray."""
    register_env("HydraRedEnv", lambda config: HydraRedEnv(config))

def create_trainer(config):
    """Create and configure an RLlib trainer."""
    trainer_config = {
        "env": "HydraRedEnv",
        "framework": "torch",
        "num_workers": 1,
        "train_batch_size": config.get("agent", {}).get("train_batch_size", 4000),
        "sgd_minibatch_size": config.get("agent", {}).get("sgd_minibatch_size", 128),
        "lr": config.get("agent", {}).get("learning_rate", 0.0003),
        "gamma": config.get("agent", {}).get("gamma", 0.99),
        "lambda": config.get("agent", {}).get("lambda", 0.95),
        "kl_coeff": config.get("agent", {}).get("kl_coeff", 0.2),
        "clip_param": config.get("agent", {}).get("clip_param", 0.3),
        "vf_clip_param": config.get("agent", {}).get("vf_clip_param", 10.0),
        "entropy_coeff": config.get("agent", {}).get("entropy_coeff", 0.01),
        "num_sgd_iter": config.get("agent", {}).get("num_sgd_iter", 10),
        "env_config": config.get("env", {})
    }
    
    return PPOTrainer(config=trainer_config)

def train(config_path="config.yaml", num_iterations=100):
    """Train the Red Agent."""
    # Load configuration
    config = load_config(config_path)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register environments
    register_environments()
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Training loop
    for i in range(num_iterations):
        result = trainer.train()
        logger.info(f"Iteration {i}: reward={result['episode_reward_mean']}")
    
    # Save the trained model
    checkpoint_path = trainer.save()
    logger.info(f"Model saved at: {checkpoint_path}")
    
    return trainer, checkpoint_path

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    
    # Train the agent
    trainer, checkpoint_path = train(config_path, num_iterations=10)
