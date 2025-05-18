import networkx as nx
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkNode:
    """
    Represents a node in the network graph.
    """
    def __init__(self, node_id, node_type="host", services=None, vulnerabilities=None):
        """
        Initialize a network node.
        
        Args:
            node_id (int): Unique identifier for the node
            node_type (str): Type of node (host, router, firewall, etc.)
            services (dict): Dictionary of services running on the node
            vulnerabilities (list): List of vulnerabilities present on the node
        """
        self.node_id = node_id
        self.node_type = node_type
        self.services = services or {}
        self.vulnerabilities = vulnerabilities or []
        self.compromised = False
        self.patched = False
        self.firewall_rules = []
        
        # Node attributes
        self.attributes = {
            "os": random.choice(["Windows", "Linux", "macOS"]),
            "patch_level": random.randint(1, 10),
            "importance": random.randint(1, 10)  # How critical this node is
        }
        
        logger.debug(f"Created node {node_id} of type {node_type}")
    
    def add_service(self, service_name, port, version=None):
        """Add a service to the node."""
        self.services[service_name] = {
            "port": port,
            "version": version,
            "running": True
        }
    
    def add_vulnerability(self, vuln_id, severity, description=None):
        """Add a vulnerability to the node."""
        self.vulnerabilities.append({
            "id": vuln_id,
            "severity": severity,
            "description": description,
            "exploited": False
        })
    
    def compromise(self):
        """Mark the node as compromised."""
        if not self.compromised:
            self.compromised = True
            logger.info(f"Node {self.node_id} has been compromised")
            return True
        return False
    
    def patch(self, vuln_id=None):
        """
        Apply a patch to the node.
        
        Args:
            vuln_id (str): If provided, patch only this vulnerability
        """
        if vuln_id:
            for vuln in self.vulnerabilities:
                if vuln["id"] == vuln_id:
                    self.vulnerabilities.remove(vuln)
                    logger.info(f"Vulnerability {vuln_id} patched on node {self.node_id}")
                    return True
            return False
        else:
            # Patch all vulnerabilities
            self.vulnerabilities = []
            self.patched = True
            logger.info(f"All vulnerabilities patched on node {self.node_id}")
            return True
    
    def add_firewall_rule(self, rule):
        """Add a firewall rule to the node."""
        self.firewall_rules.append(rule)
        logger.info(f"Firewall rule added to node {self.node_id}: {rule}")
    
    def to_dict(self):
        """Convert node to dictionary for serialization."""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "services": self.services,
            "vulnerabilities": self.vulnerabilities,
            "compromised": self.compromised,
            "patched": self.patched,
            "firewall_rules": self.firewall_rules,
            "attributes": self.attributes
        }

class GraphEnvironment:
    """
    Graph-based environment for HYDRA using NetworkX.
    Represents the network topology as a graph.
    """
    def __init__(self, num_nodes=10, connectivity=0.3, seed=None):
        """
        Initialize the graph environment.
        
        Args:
            num_nodes (int): Number of nodes in the network
            connectivity (float): Probability of edge creation between nodes
            seed (int): Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.num_nodes = num_nodes
        self.connectivity = connectivity
        
        # Create the graph
        self.graph = nx.erdos_renyi_graph(n=num_nodes, p=connectivity, seed=seed)
        
        # Create node objects
        self.nodes = {}
        for node_id in self.graph.nodes():
            node_type = self._assign_node_type(node_id)
            self.nodes[node_id] = NetworkNode(node_id, node_type)
            
            # Add random services and vulnerabilities
            self._add_random_services(node_id)
            self._add_random_vulnerabilities(node_id)
        
        # Entry point for attacks (usually the internet-facing node)
        self.entry_node = 0
        
        # Track compromised nodes
        self.compromised_nodes = set()
        
        # Track logs
        self.logs = []
        
        # Track actions taken
        self.actions_history = []
        
        logger.info(f"Created graph environment with {num_nodes} nodes and {self.graph.number_of_edges()} connections")
    
    def _assign_node_type(self, node_id):
        """Assign a type to a node based on its position in the network."""
        if node_id == 0:
            return "internet_gateway"
        elif node_id == self.num_nodes - 1:
            return "database"
        elif node_id % 3 == 0:
            return "web_server"
        elif node_id % 3 == 1:
            return "workstation"
        else:
            return "internal_server"
    
    def _add_random_services(self, node_id):
        """Add random services to a node based on its type."""
        node = self.nodes[node_id]
        
        if node.node_type == "internet_gateway":
            node.add_service("http", 80)
            node.add_service("https", 443)
            node.add_service("ssh", 22)
        
        elif node.node_type == "web_server":
            node.add_service("http", 80)
            node.add_service("https", 443)
            node.add_service("ssh", 22)
            node.add_service("database", 5432)
        
        elif node.node_type == "database":
            node.add_service("database", 5432)
            node.add_service("ssh", 22)
        
        elif node.node_type == "workstation":
            node.add_service("smb", 445)
            if random.random() < 0.3:
                node.add_service("rdp", 3389)
        
        else:  # internal_server
            services = ["http", "https", "ssh", "smb", "ldap", "dns"]
            ports = [80, 443, 22, 445, 389, 53]
            
            # Add 2-4 random services
            num_services = random.randint(2, 4)
            for _ in range(num_services):
                idx = random.randint(0, len(services) - 1)
                node.add_service(services[idx], ports[idx])
    
    def _add_random_vulnerabilities(self, node_id):
        """Add random vulnerabilities to a node based on its services."""
        node = self.nodes[node_id]
        
        # Vulnerability types
        vuln_types = [
            ("CVE-2021-1234", "Remote Code Execution in Web Server"),
            ("CVE-2020-5678", "SQL Injection in Database"),
            ("CVE-2019-9012", "Privilege Escalation"),
            ("CVE-2022-3456", "Buffer Overflow"),
            ("CVE-2018-7890", "Authentication Bypass")
        ]
        
        # Add 0-3 random vulnerabilities
        num_vulns = random.randint(0, 3)
        for _ in range(num_vulns):
            vuln = random.choice(vuln_types)
            severity = random.randint(1, 10)
            node.add_vulnerability(vuln[0], severity, vuln[1])
    
    def get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            np.array: State representation
        """
        # Create a state vector:
        # - First part: Binary indicators for compromised nodes
        # - Second part: Binary indicators for patched nodes
        # - Third part: Count of vulnerabilities per node
        
        compromised = np.zeros(self.num_nodes)
        patched = np.zeros(self.num_nodes)
        vuln_counts = np.zeros(self.num_nodes)
        
        for node_id, node in self.nodes.items():
            compromised[node_id] = 1 if node.compromised else 0
            patched[node_id] = 1 if node.patched else 0
            vuln_counts[node_id] = len(node.vulnerabilities)
        
        return np.concatenate([compromised, patched, vuln_counts])
    
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            np.array: Initial state
        """
        # Reset all nodes
        for node in self.nodes.values():
            node.compromised = False
            node.patched = False
        
        self.compromised_nodes = set()
        self.logs = []
        self.actions_history = []
        
        return self.get_state()
    
    def attack_node(self, node_id, attack_type="exploit"):
        """
        Attempt to attack a node.
        
        Args:
            node_id (int): ID of the node to attack
            attack_type (str): Type of attack
            
        Returns:
            tuple: (success, reward, log_message)
        """
        if node_id not in self.graph.nodes():
            return False, -1, f"Invalid node ID: {node_id}"
        
        node = self.nodes[node_id]
        
        # Check if the node is already compromised
        if node.compromised:
            return False, 0, f"Node {node_id} is already compromised"
        
        # Check if the node is reachable from a compromised node
        reachable = False
        if not self.compromised_nodes:
            # If no nodes are compromised yet, only the entry node is reachable
            reachable = (node_id == self.entry_node)
        else:
            # Check if there's a path from any compromised node
            for comp_node in self.compromised_nodes:
                if nx.has_path(self.graph, comp_node, node_id):
                    reachable = True
                    break
        
        if not reachable:
            return False, -1, f"Node {node_id} is not reachable"
        
        # Check if the node has vulnerabilities
        if not node.vulnerabilities:
            return False, -0.5, f"Node {node_id} has no vulnerabilities"
        
        # Attempt to exploit
        success_prob = 0.7 if attack_type == "exploit" else 0.4
        success = random.random() < success_prob
        
        if success:
            node.compromise()
            self.compromised_nodes.add(node_id)
            
            # Higher reward for more important nodes
            reward = 1.0 + (node.attributes["importance"] / 10.0)
            log_message = f"Successfully compromised node {node_id} ({node.node_type})"
            
            # Mark a random vulnerability as exploited
            if node.vulnerabilities:
                vuln = random.choice(node.vulnerabilities)
                vuln["exploited"] = True
        else:
            reward = -0.2
            log_message = f"Failed to compromise node {node_id}"
        
        # Record the action
        self.actions_history.append({
            "type": "attack",
            "node_id": node_id,
            "attack_type": attack_type,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add to logs
        self.logs.append(log_message)
        
        return success, reward, log_message
    
    def defend_node(self, node_id, defense_type="patch"):
        """
        Attempt to defend a node.
        
        Args:
            node_id (int): ID of the node to defend
            defense_type (str): Type of defense (patch, firewall, etc.)
            
        Returns:
            tuple: (success, reward, log_message)
        """
        if node_id not in self.graph.nodes():
            return False, -1, f"Invalid node ID: {node_id}"
        
        node = self.nodes[node_id]
        
        if defense_type == "patch":
            # Attempt to patch vulnerabilities
            if not node.vulnerabilities:
                return False, -0.5, f"Node {node_id} has no vulnerabilities to patch"
            
            success = node.patch()
            
            if success:
                # If the node was compromised, it's now secured
                if node.compromised:
                    node.compromised = False
                    if node_id in self.compromised_nodes:
                        self.compromised_nodes.remove(node_id)
                
                reward = 1.0
                log_message = f"Successfully patched node {node_id}"
            else:
                reward = -0.2
                log_message = f"Failed to patch node {node_id}"
        
        elif defense_type == "firewall":
            # Add a firewall rule
            rule = f"block_all_incoming_{int(time.time())}"
            node.add_firewall_rule(rule)
            
            # This makes it harder to compromise the node
            success = True
            reward = 0.5
            log_message = f"Added firewall rule to node {node_id}"
        
        else:
            return False, -1, f"Unknown defense type: {defense_type}"
        
        # Record the action
        self.actions_history.append({
            "type": "defense",
            "node_id": node_id,
            "defense_type": defense_type,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add to logs
        self.logs.append(log_message)
        
        return success, reward, log_message
    
    def visualize(self, save_path=None):
        """
        Visualize the current state of the network.
        
        Args:
            save_path (str): If provided, save the visualization to this path
        """
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(self.graph)
        
        # Node colors based on status
        node_colors = []
        for node_id in self.graph.nodes():
            node = self.nodes[node_id]
            if node.compromised:
                node_colors.append('red')
            elif node.patched:
                node_colors.append('green')
            else:
                node_colors.append('blue')
        
        # Draw the network
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_color='white'
        )
        
        # Add a title
        plt.title("Network Topology - Red: Compromised, Green: Patched, Blue: Normal")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Network visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_logs(self):
        """Get all logs."""
        return self.logs
    
    def get_actions_history(self):
        """Get the history of actions taken."""
        return self.actions_history
    
    def get_network_stats(self):
        """Get statistics about the network."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.graph.number_of_edges(),
            "compromised_nodes": len(self.compromised_nodes),
            "patched_nodes": sum(1 for node in self.nodes.values() if node.patched),
            "total_vulnerabilities": sum(len(node.vulnerabilities) for node in self.nodes.values())
        }

# Example usage
if __name__ == "__main__":
    # Create a graph environment
    env = GraphEnvironment(num_nodes=10, connectivity=0.3)
    
    # Visualize the initial state
    env.visualize("initial_network.png")
    
    # Perform some attacks
    env.attack_node(0, "exploit")
    env.attack_node(1, "exploit")
    
    # Perform some defenses
    env.defend_node(2, "patch")
    
    # Visualize the final state
    env.visualize("final_network.png")
    
    # Print logs
    for log in env.get_logs():
        print(log)
