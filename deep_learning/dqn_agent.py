import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """
    Deep Q-Network model for HYDRA agents.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32]):
        super(DQN, self).__init__()

        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)
        logger.info(f"Created DQN with architecture: {input_dim} -> {hidden_dims} -> {output_dim}")

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    """
    DQN Agent implementation for HYDRA.
    Can be used for both Red and Blue agents.
    """
    def __init__(self, state_size, action_size, agent_type="red", config=None):
        """
        Initialize the DQN Agent.

        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
            agent_type (str): "red" or "blue"
            config (dict): Configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_type = agent_type

        # Set default config if not provided
        if config is None:
            config = {}

        # Extract configuration parameters with defaults
        self.memory_size = config.get("memory_size", 10000)
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.95)  # Discount factor
        self.epsilon = config.get("epsilon_start", 1.0)  # Exploration rate
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.target_update_freq = config.get("target_update_freq", 10)  # Update target network every N steps
        self.hidden_dims = config.get("hidden_dims", [64, 32])

        # Initialize networks
        self.policy_net = DQN(state_size, action_size, self.hidden_dims)
        self.target_net = DQN(state_size, action_size, self.hidden_dims)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is used for evaluation only

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Initialize replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Training stats
        self.steps_done = 0
        self.episodes_done = 0

        logger.info(f"Initialized {agent_type.upper()} DQN Agent with state_size={state_size}, action_size={action_size}")

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, eval_mode=False):
        """
        Choose an action based on the current state.

        Args:
            state: Current state
            eval_mode (bool): If True, use greedy policy (no exploration)

        Returns:
            int: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randrange(self.action_size)
        else:
            # Exploit: choose best action according to policy network
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def replay(self):
        """
        Train the network using experiences from replay memory.
        """
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples

        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)

        # Compute next Q values using target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filepath):
        """
        Save the model to disk.
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load the model from disk.
        """
        if not torch.cuda.is_available():
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(filepath)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        logger.info(f"Model loaded from {filepath}")

class RedDQNAgent(DQNAgent):
    """
    DQN Agent specialized for the Red Team (attacker).
    """
    def __init__(self, state_size, action_size, config=None):
        super().__init__(state_size, action_size, agent_type="red", config=config)

        # Red-specific parameters
        if config is None:
            config = {}
        self.attack_success_threshold = config.get("attack_success_threshold", 0.7)

        logger.info("Initialized Red DQN Agent with attack specialization")

class BlueDQNAgent(DQNAgent):
    """
    DQN Agent specialized for the Blue Team (defender).
    """
    def __init__(self, state_size, action_size, config=None):
        super().__init__(state_size, action_size, agent_type="blue", config=config)

        # Blue-specific parameters
        if config is None:
            config = {}
        self.defense_threshold = config.get("defense_threshold", 0.6)

        logger.info("Initialized Blue DQN Agent with defense specialization")
