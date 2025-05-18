# 🐉 Project HYDRA – The Self-Evolving Red-Blue AI Symbiote

<p align="center">
  <img src="https://github.com/username/project-hydra/raw/main/docs/images/hydra-logo.png" alt="HYDRA Logo" width="200"/>
</p>

<p align="center">
  <em>"Each head attacks; the others defend—co‑training in lockstep to forge undefeatable security."</em>
</p>

## Overview
Project HYDRA is a self-evolving red-blue AI symbiote for cybersecurity. This repository contains the Phase 1 implementation, which focuses on creating a sandboxed digital twin of a target web service and implementing Red & Blue agents to explore attack/defense dynamics.

HYDRA uses reinforcement learning and graph-based environments to simulate cyber attacks and defenses, creating a continuous improvement loop where both offensive and defensive capabilities evolve together.

## Core Concept
HYDRA uses a GAN-style approach where:
- **Red Agent**: Discovers and exploits vulnerabilities in the digital twin
- **Blue Agent**: Detects attacks and implements defenses
- **Shared Replay Buffer**: Facilitates co-training between agents
- **Digital Twin**: Provides a safe environment for agents to operate

## Core Innovations

### Digital Twin Co-Simulation
- Full-stack "mirror" of your network, applications, and users in a sandbox
- Both Red Agent (attacker) and Blue Agent (defender) operate in parallel on this twin

### Adversarial Co-Training Loop
- GAN-style approach: the Red Agent's novel exploits are used as "adversarial examples" to train the Blue Agent's defenses
- Defenses hardened against the latest AI-generated attack vectors in real time

### Graph Neural Mapping
- Models entire enterprise as a heterogeneous graph (hosts, users, processes, data flows)
- Finds not just shortest attack paths but novel lateral movement routes

## Demo

<p align="center">
  <img src="https://github.com/username/project-hydra/raw/main/docs/images/network_visualization.png" alt="Network Visualization" width="600"/>
</p>

## Phase 1 Goals
- ✅ Set up a simple VM cluster replicating a target web app + database
- ✅ Build Red & Blue agents using both RLlib and DQN approaches
- ✅ Implement basic actions: port scan, brute-force (Red), firewall block (Blue)
- ✅ Establish the co-training loop foundation
- ✅ Create a graph-based network environment
- ✅ Implement a dashboard for monitoring training progress

## Prerequisites
- Python 3.8+
- PyTorch
- NetworkX
- Flask (for dashboard)
- Docker & Docker Compose (optional, for full digital twin)

## Project Structure
```
project-hydra/
├── README.md                # Overview, setup & run instructions
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Orchestrates the Digital Twin sandbox
├── Dockerfile               # Main application Dockerfile
├── twin_env/                # Digital Twin definition
│   ├── docker-compose.yml   # Web app + DB stack
│   └── app/
│       ├── Dockerfile       # Simple Flask app image
│       └── app.py           # Minimal target web service with vulnerabilities
├── red_agent/               # Red Agent (attacker)
│   ├── __init__.py
│   ├── red_agent.py         # RLlib-based agent loop (scan→attack→log)
│   └── config.yaml          # RL hyperparameters & action definitions
├── blue_agent/              # Blue Agent (defender)
│   ├── __init__.py
│   ├── blue_agent.py        # RLlib-based defender loop (monitor→block→patch)
│   └── config.yaml          # Hyperparameters & defense actions
├── deep_learning/           # Deep learning components
│   └── dqn_agent.py         # DQN implementation for both agents
├── graph_env/               # Graph-based environment
│   └── graph_env.py         # NetworkX-based network simulation
├── dashboard/               # Web dashboard
│   ├── app.py               # Flask dashboard application
│   └── templates/
│       └── index.html       # Dashboard UI
├── shared/
│   └── replay_buffer.py     # Shared buffer for co-training and logs
├── main.py                  # Original launcher with RLlib agents
└── main_enhanced.py         # Enhanced launcher with DQN, graph env, and dashboard
```

## Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Launch the twin environment
docker-compose up -d

# 3. Run training with enhanced features
python main_enhanced.py --use-graph --use-dqn --dashboard

# Or run the original training
python main.py
```

## Components

### Digital Twin
A sandboxed environment that mirrors a real-world application stack, containing intentional vulnerabilities for the Red Agent to discover.

### Red Agent
An AI-based attacker that learns to:
- Scan for open ports and services
- Attempt brute-force attacks
- Exploit known vulnerabilities
- Log successful exploits to the shared replay buffer

### Blue Agent
An AI-based defender that learns to:
- Monitor system logs and network traffic
- Detect anomalous behavior
- Implement firewall rules to block attacks
- Log successful defenses to the shared replay buffer

### Graph Environment
A network simulation that:
- Models the target infrastructure as a graph
- Provides a realistic topology for attack and defense
- Tracks node compromises and patches
- Visualizes the network state

### DQN Agents
Deep Q-Network implementation that:
- Provides an alternative to RLlib agents
- Uses PyTorch for deep learning
- Implements experience replay and target networks
- Specializes for Red and Blue team operations

### Dashboard
A web-based monitoring interface that:
- Displays training progress in real-time
- Visualizes the network topology
- Shows agent actions and rewards
- Logs system events

### Shared Replay Buffer
A common data structure that:
- Stores successful attacks and defenses
- Provides training examples for both agents
- Facilitates the adversarial co-training loop

## Command Line Options
```
python main_enhanced.py --help
```

- `--iterations`: Number of training iterations (default: 10)
- `--red-config`: Path to Red Agent configuration
- `--blue-config`: Path to Blue Agent configuration
- `--skip-twin`: Skip digital twin setup/teardown
- `--use-dqn`: Use DQN agents instead of RLlib
- `--use-graph`: Use graph-based environment
- `--dashboard`: Enable dashboard

## Next Steps
- Expand action spaces for both agents
- Implement more sophisticated reward functions
- Add more realistic vulnerabilities to the twin environment
- Develop metrics for evaluating agent performance
- Implement the Generative Exploit Module (Phase 2)
- Enhance the adversarial co-training loop (Phase 3)
