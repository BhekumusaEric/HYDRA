import numpy as np
import logging
import time
import json
import os
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    Shared replay buffer for storing experiences from both Red and Blue agents.
    This facilitates the adversarial co-training loop.
    """
    
    def __init__(self, max_size=10000, save_dir="./buffer_data"):
        """
        Initialize the replay buffer.
        
        Args:
            max_size (int): Maximum number of experiences to store
            save_dir (str): Directory to save buffer data
        """
        self.red_buffer = deque(maxlen=max_size)
        self.blue_buffer = deque(maxlen=max_size)
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        logger.info(f"Replay buffer initialized with max size {max_size}")
    
    def add_red_experience(self, experience):
        """
        Add a Red Agent experience to the buffer.
        
        Args:
            experience (dict): Experience data including state, action, reward, next_state, done
        """
        self.red_buffer.append(experience)
        
        # If this was a successful exploit, also add it to the Blue buffer as a training example
        if experience.get("exploit_success", False):
            blue_experience = {
                "attack_type": experience.get("action_info", {}).get("type", "unknown"),
                "target": experience.get("action_info", {}).get("target", "unknown"),
                "timestamp": time.time(),
                "state": experience.get("state"),
                "exploit_details": experience.get("action_info")
            }
            self.blue_buffer.append(blue_experience)
            logger.info(f"Added successful exploit to Blue buffer: {blue_experience['attack_type']} on {blue_experience['target']}")
    
    def add_blue_experience(self, experience):
        """
        Add a Blue Agent experience to the buffer.
        
        Args:
            experience (dict): Experience data including state, action, reward, next_state, done
        """
        self.blue_buffer.append(experience)
        
        # If this was a successful defense, also add it to the Red buffer as a training example
        if experience.get("defense_success", False):
            red_experience = {
                "defense_type": experience.get("action_info", {}).get("type", "unknown"),
                "target": experience.get("action_info", {}).get("target", "unknown"),
                "timestamp": time.time(),
                "state": experience.get("state"),
                "defense_details": experience.get("action_info")
            }
            self.red_buffer.append(red_experience)
            logger.info(f"Added successful defense to Red buffer: {red_experience['defense_type']} on {red_experience['target']}")
    
    def sample_red_batch(self, batch_size=32):
        """
        Sample a batch of experiences from the Red buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: Batch of experiences
        """
        if len(self.red_buffer) < batch_size:
            return list(self.red_buffer)
        
        return np.random.choice(list(self.red_buffer), batch_size, replace=False)
    
    def sample_blue_batch(self, batch_size=32):
        """
        Sample a batch of experiences from the Blue buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: Batch of experiences
        """
        if len(self.blue_buffer) < batch_size:
            return list(self.blue_buffer)
        
        return np.random.choice(list(self.blue_buffer), batch_size, replace=False)
    
    def save(self):
        """Save the buffer data to disk."""
        timestamp = int(time.time())
        
        # Save Red buffer
        red_path = os.path.join(self.save_dir, f"red_buffer_{timestamp}.json")
        with open(red_path, 'w') as f:
            # Convert deque to list for serialization
            json.dump(list(self.red_buffer), f)
        
        # Save Blue buffer
        blue_path = os.path.join(self.save_dir, f"blue_buffer_{timestamp}.json")
        with open(blue_path, 'w') as f:
            # Convert deque to list for serialization
            json.dump(list(self.blue_buffer), f)
        
        logger.info(f"Saved buffer data to {self.save_dir}")
        return red_path, blue_path
    
    def load(self, red_path, blue_path):
        """
        Load buffer data from disk.
        
        Args:
            red_path (str): Path to Red buffer data
            blue_path (str): Path to Blue buffer data
        """
        # Load Red buffer
        if os.path.exists(red_path):
            with open(red_path, 'r') as f:
                red_data = json.load(f)
                self.red_buffer = deque(red_data, maxlen=self.red_buffer.maxlen)
        
        # Load Blue buffer
        if os.path.exists(blue_path):
            with open(blue_path, 'r') as f:
                blue_data = json.load(f)
                self.blue_buffer = deque(blue_data, maxlen=self.blue_buffer.maxlen)
        
        logger.info(f"Loaded buffer data: {len(self.red_buffer)} red experiences, {len(self.blue_buffer)} blue experiences")
    
    def get_stats(self):
        """
        Get statistics about the buffer.
        
        Returns:
            dict: Buffer statistics
        """
        return {
            "red_buffer_size": len(self.red_buffer),
            "blue_buffer_size": len(self.blue_buffer),
            "red_exploit_types": self._count_exploit_types(),
            "blue_defense_types": self._count_defense_types()
        }
    
    def _count_exploit_types(self):
        """Count the types of exploits in the Red buffer."""
        exploit_types = {}
        for exp in self.red_buffer:
            if "action_info" in exp and "type" in exp["action_info"]:
                exploit_type = exp["action_info"]["type"]
                exploit_types[exploit_type] = exploit_types.get(exploit_type, 0) + 1
        return exploit_types
    
    def _count_defense_types(self):
        """Count the types of defenses in the Blue buffer."""
        defense_types = {}
        for exp in self.blue_buffer:
            if "action_info" in exp and "type" in exp["action_info"]:
                defense_type = exp["action_info"]["type"]
                defense_types[defense_type] = defense_types.get(defense_type, 0) + 1
        return defense_types

# Global instance for easy access
buffer = ReplayBuffer()

def get_buffer():
    """Get the global buffer instance."""
    return buffer
