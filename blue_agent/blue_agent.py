import os
import yaml
import gym
import numpy as np
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

class HydraBlueEnv(gym.Env):
    """
    Environment for the Blue Agent in Project HYDRA.
    This environment simulates defending a web application from attacks.
    """
    
    def __init__(self, config=None):
        config = config or {}
        
        # Load configuration
        self.target_host = config.get("target_host", "web-app")
        self.target_ports = config.get("target_ports", [5000, 5432])
        self.max_steps = config.get("max_steps_per_episode", 100)
        
        # Set up the action space
        # Actions: 
        # 0-4: Block IP addresses
        # 5-6: Block ports
        # 7-9: Rate limit endpoints
        # 10-12: Add input validation
        # 13-15: Increase monitoring
        self.action_space = spaces.Discrete(16)
        
        # Set up the observation space
        # Observations include:
        # - Login attempts (count, success rate)
        # - Suspicious requests (count by type)
        # - Currently blocked IPs and ports
        # - Endpoint traffic statistics
        self.observation_space = spaces.Dict({
            "login_attempts": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),  # [count, success_rate]
            "suspicious_requests": spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32),  # [sql_injection, path_traversal, other]
            "blocked_resources": spaces.MultiBinary(10),  # [ip1, ip2, ip3, ip4, ip5, port1, port2, port3, port4, port5]
            "endpoint_traffic": spaces.Box(low=0, high=1000, shape=(4,), dtype=np.float32),  # [login, user_info, download, health]
            "step_count": spaces.Discrete(self.max_steps + 1)
        })
        
        # Initialize state
        self.reset()
        
        logger.info("HydraBlueEnv initialized")
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        
        # Initialize state variables
        self.login_attempts = np.array([0, 0.5], dtype=np.float32)  # [count, success_rate]
        self.suspicious_requests = np.zeros(3, dtype=np.float32)  # [sql_injection, path_traversal, other]
        self.blocked_resources = np.zeros(10, dtype=np.int8)  # [ip1, ip2, ip3, ip4, ip5, port1, port2, port3, port4, port5]
        self.endpoint_traffic = np.zeros(4, dtype=np.float32)  # [login, user_info, download, health]
        
        # Attack simulation state
        self.current_attacks = []
        self.attack_success_count = 0
        self.legitimate_requests_blocked = 0
        
        # Generate initial observation
        obs = self._get_observation()
        
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
        
        # Simulate incoming traffic and attacks
        self._simulate_traffic()
        
        # Process the action
        if action < 5:
            # Block IP address
            ip_idx = action
            success, legitimate_blocked = self._block_ip(ip_idx)
            if success:
                self.blocked_resources[ip_idx] = 1
                reward += 2.0
                info["block_ip"] = f"Successfully blocked attacking IP {ip_idx}"
            elif legitimate_blocked:
                self.blocked_resources[ip_idx] = 1
                reward -= 1.0
                info["block_ip"] = f"Blocked legitimate IP {ip_idx}"
            else:
                info["block_ip"] = f"No effect from blocking IP {ip_idx}"
                
        elif action < 7:
            # Block port
            port_idx = action - 5
            success, legitimate_blocked = self._block_port(port_idx)
            if success:
                self.blocked_resources[5 + port_idx] = 1
                reward += 2.0
                info["block_port"] = f"Successfully blocked attacking port {self.target_ports[port_idx]}"
            elif legitimate_blocked:
                self.blocked_resources[5 + port_idx] = 1
                reward -= 1.0
                info["block_port"] = f"Blocked legitimate traffic on port {self.target_ports[port_idx]}"
            else:
                info["block_port"] = f"No effect from blocking port {self.target_ports[port_idx]}"
        
        elif action < 10:
            # Rate limit endpoint
            endpoint_idx = action - 7
            endpoints = ["login", "user_info", "download"]
            if endpoint_idx < len(endpoints):
                endpoint = endpoints[endpoint_idx]
                success, legitimate_blocked = self._rate_limit_endpoint(endpoint)
                if success:
                    reward += 0.5
                    info["rate_limit"] = f"Successfully rate-limited attacking traffic to {endpoint}"
                elif legitimate_blocked:
                    reward -= 0.5
                    info["rate_limit"] = f"Rate-limited legitimate traffic to {endpoint}"
                else:
                    info["rate_limit"] = f"No effect from rate-limiting {endpoint}"
        
        elif action < 13:
            # Add input validation
            endpoint_idx = action - 10
            endpoints = ["login", "user_info", "download"]
            if endpoint_idx < len(endpoints):
                endpoint = endpoints[endpoint_idx]
                success = self._add_input_validation(endpoint)
                if success:
                    reward += 1.0
                    info["input_validation"] = f"Successfully added input validation to {endpoint}"
                else:
                    info["input_validation"] = f"No effect from adding input validation to {endpoint}"
        
        else:
            # Increase monitoring
            endpoint_idx = action - 13
            endpoints = ["login", "user_info", "download"]
            if endpoint_idx < len(endpoints):
                endpoint = endpoints[endpoint_idx]
                detected = self._increase_monitoring(endpoint)
                if detected:
                    reward += 0.5
                    info["monitoring"] = f"Detected attack on {endpoint} through increased monitoring"
                else:
                    info["monitoring"] = f"No attacks detected on {endpoint}"
        
        # Check for undetected attacks
        undetected_attacks = self._check_undetected_attacks()
        if undetected_attacks:
            reward -= 2.0 * undetected_attacks
            info["undetected_attacks"] = f"{undetected_attacks} attacks went undetected"
        
        # Get updated observation
        obs = self._get_observation()
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Construct the current observation."""
        return {
            "login_attempts": self.login_attempts,
            "suspicious_requests": self.suspicious_requests,
            "blocked_resources": self.blocked_resources,
            "endpoint_traffic": self.endpoint_traffic,
            "step_count": self.current_step
        }
    
    def _simulate_traffic(self):
        """Simulate incoming traffic and attacks."""
        # Simulate legitimate traffic
        self.endpoint_traffic += np.random.poisson(lam=[5, 3, 2, 1], size=4)
        
        # Simulate login attempts
        new_login_attempts = np.random.poisson(lam=3)
        success_rate = 0.7  # 70% of legitimate logins succeed
        self.login_attempts[0] += new_login_attempts
        self.login_attempts[1] = (self.login_attempts[1] * (self.login_attempts[0] - new_login_attempts) + 
                                 success_rate * new_login_attempts) / self.login_attempts[0]
        
        # Simulate attacks with some probability
        if np.random.random() < 0.3:  # 30% chance of attack per step
            attack_type = np.random.choice(["sql_injection", "path_traversal", "brute_force"])
            target_endpoint = np.random.choice(["login", "user_info", "download"])
            
            # Record the attack
            self.current_attacks.append({
                "type": attack_type,
                "target": target_endpoint,
                "detected": False,
                "blocked": False
            })
            
            # Update observation based on attack type
            if attack_type == "sql_injection":
                self.suspicious_requests[0] += 1
                if target_endpoint == "user_info":
                    self.endpoint_traffic[1] += 1
            elif attack_type == "path_traversal":
                self.suspicious_requests[1] += 1
                if target_endpoint == "download":
                    self.endpoint_traffic[2] += 1
            elif attack_type == "brute_force":
                self.suspicious_requests[2] += 1
                if target_endpoint == "login":
                    self.login_attempts[0] += 10  # Brute force generates many login attempts
                    self.login_attempts[1] = (self.login_attempts[1] * (self.login_attempts[0] - 10)) / self.login_attempts[0]  # Reduce success rate
                    self.endpoint_traffic[0] += 10
    
    def _block_ip(self, ip_idx):
        """Simulate blocking an IP address."""
        # Check if there are any attacks from this IP
        has_attack = False
        legitimate_blocked = False
        
        for attack in self.current_attacks:
            if not attack["blocked"] and np.random.randint(0, 5) == ip_idx:  # Randomly assign attack to an IP
                attack["blocked"] = True
                has_attack = True
        
        # Check if we're blocking legitimate traffic
        if not has_attack and np.random.random() < 0.5:  # 50% chance of blocking legitimate traffic if no attack
            legitimate_blocked = True
        
        return has_attack, legitimate_blocked
    
    def _block_port(self, port_idx):
        """Simulate blocking a port."""
        # Check if there are any attacks on this port
        has_attack = False
        legitimate_blocked = False
        
        for attack in self.current_attacks:
            if not attack["blocked"]:
                if (port_idx == 0 and attack["target"] in ["login", "user_info", "download"]) or \
                   (port_idx == 1 and attack["type"] == "database_attack"):
                    attack["blocked"] = True
                    has_attack = True
        
        # Check if we're blocking legitimate traffic
        if np.random.random() < 0.7:  # 70% chance of blocking legitimate traffic (ports have a lot of legitimate traffic)
            legitimate_blocked = True
        
        return has_attack, legitimate_blocked
    
    def _rate_limit_endpoint(self, endpoint):
        """Simulate rate limiting an endpoint."""
        endpoint_map = {"login": 0, "user_info": 1, "download": 2}
        endpoint_idx = endpoint_map.get(endpoint, 0)
        
        has_attack = False
        legitimate_blocked = False
        
        for attack in self.current_attacks:
            if not attack["blocked"] and attack["target"] == endpoint and attack["type"] == "brute_force":
                attack["blocked"] = True
                has_attack = True
        
        # Check if we're blocking legitimate traffic
        if self.endpoint_traffic[endpoint_idx] > 10 and np.random.random() < 0.3:  # 30% chance of affecting legitimate traffic if high volume
            legitimate_blocked = True
        
        return has_attack, legitimate_blocked
    
    def _add_input_validation(self, endpoint):
        """Simulate adding input validation to an endpoint."""
        has_effect = False
        
        for attack in self.current_attacks:
            if not attack["blocked"] and attack["target"] == endpoint and attack["type"] in ["sql_injection", "path_traversal"]:
                attack["blocked"] = True
                has_effect = True
        
        return has_effect
    
    def _increase_monitoring(self, endpoint):
        """Simulate increasing monitoring on an endpoint."""
        detected_attack = False
        
        for attack in self.current_attacks:
            if not attack["detected"] and attack["target"] == endpoint:
                attack["detected"] = True
                detected_attack = True
        
        return detected_attack
    
    def _check_undetected_attacks(self):
        """Check for attacks that weren't detected or blocked."""
        undetected_count = 0
        
        for attack in self.current_attacks:
            if not attack["detected"] and not attack["blocked"]:
                undetected_count += 1
                self.attack_success_count += 1
        
        # Clear the current attacks list for the next step
        self.current_attacks = []
        
        return undetected_count

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
    register_env("HydraBlueEnv", lambda config: HydraBlueEnv(config))

def create_trainer(config):
    """Create and configure an RLlib trainer."""
    trainer_config = {
        "env": "HydraBlueEnv",
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
    """Train the Blue Agent."""
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
