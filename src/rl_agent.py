"""
Reinforcement Learning Agent for Valet Parking Game.
This module implements a Deep Q-Learning agent that can learn to:
1. Park cars in their designated spots
2. Return cars to their respective clients
"""

import numpy as np
import random
import pygame
import copy
from collections import deque
import os
import json

# Try importing torch, but provide fallback to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using NumPy-based implementation instead.")

# Import game-related modules 
from .config import *

# Constants for RL
MEMORY_SIZE = 100000  # Experience replay buffer size
GAMMA = 0.99         # Discount factor
EPSILON = 1.0        # Exploration rate
EPSILON_MIN = 0.1    # Minimum exploration probability
EPSILON_DECAY = 0.995  # Exponential decay rate for exploration
LEARNING_RATE = 0.001  # Learning rate for optimizer
BATCH_SIZE = 64      # Size of minibatch from memory
UPDATE_TARGET_FREQUENCY = 1000  # How often to update target network
TRAINING_START = 5000  # Start training after this many steps

# Observation and action space dimensions
# State: [player_x, player_y, car1_x, car1_y, car1_angle, car1_parked, 
#        car2_x, car2_y, car2_angle, car2_parked, ...]
# + [client1_x, client1_y, client1_car_type, client2_x, client2_y, client2_car_type, ...]
# Actions: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:SPACE, 5:NONE (do nothing)
ACTION_SPACE = 6

class DQNModel:
    """Deep Q-Network model using PyTorch or NumPy if PyTorch is not available."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        if TORCH_AVAILABLE:
            self._build_torch_model()
        else:
            self._build_numpy_model()
    
    def _build_torch_model(self):
        """Build a PyTorch neural network model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(QNetwork, self).__init__()
                
                self.layers = nn.Sequential(
                    nn.Linear(state_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_size)
                )
                
            def forward(self, x):
                return self.layers(x)
        
        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
    def _build_numpy_model(self):
        """Build a simple NumPy-based model as PyTorch fallback."""
        # Simple 3-layer neural network implementation with NumPy
        self.w1 = np.random.randn(self.state_size, 256) / np.sqrt(self.state_size)
        self.b1 = np.zeros(256)
        self.w2 = np.random.randn(256, 256) / np.sqrt(256)
        self.b2 = np.zeros(256)
        self.w3 = np.random.randn(256, self.action_size) / np.sqrt(256)
        self.b3 = np.zeros(self.action_size)
        
        # Target network weights and biases
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b3 = self.b3.copy()
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def forward(self, state):
        """Forward pass: compute Q-values for a state."""
        if TORCH_AVAILABLE:
            return self._forward_torch(state)
        else:
            return self._forward_numpy(state)
    
    def _forward_torch(self, state):
        """Forward pass using PyTorch."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            return self.model(state).cpu().numpy()
    
    def _forward_numpy(self, state):
        """Forward pass using NumPy."""
        x = state
        x = self.relu(np.dot(x, self.w1) + self.b1)
        x = self.relu(np.dot(x, self.w2) + self.b2)
        q_values = np.dot(x, self.w3) + self.b3
        return q_values
    
    def target_forward(self, state):
        """Forward pass through target network."""
        if TORCH_AVAILABLE:
            return self._target_forward_torch(state)
        else:
            return self._target_forward_numpy(state)
    
    def _target_forward_torch(self, state):
        """Forward pass using PyTorch target network."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            return self.target_model(state).cpu().numpy()
    
    def _target_forward_numpy(self, state):
        """Forward pass using NumPy target network."""
        x = state
        x = self.relu(np.dot(x, self.target_w1) + self.target_b1)
        x = self.relu(np.dot(x, self.target_w2) + self.target_b2)
        q_values = np.dot(x, self.target_w3) + self.target_b3
        return q_values
    
    def train(self, states, targets):
        """Train the model on a batch."""
        if TORCH_AVAILABLE:
            self._train_torch(states, targets)
        else:
            self._train_numpy(states, targets)
    
    def _train_torch(self, states, targets):
        """Train using PyTorch."""
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        
        # Forward pass
        predictions = self.model(states)
        
        # Compute loss
        loss = self.criterion(predictions, targets)
        
        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _train_numpy(self, states, targets):
        """Simple training step using NumPy with SGD."""
        # Learning rate for numpy implementation
        lr = 0.001
        
        # For each training example
        for i in range(len(states)):
            state = states[i]
            target = targets[i]
            
            # Forward pass
            h1 = self.relu(np.dot(state, self.w1) + self.b1)
            h2 = self.relu(np.dot(h1, self.w2) + self.b2)
            output = np.dot(h2, self.w3) + self.b3
            
            # Compute gradients (very simplified)
            delta3 = output - target
            delta2 = np.dot(delta3, self.w3.T) * (h2 > 0)
            delta1 = np.dot(delta2, self.w2.T) * (h1 > 0)
            
            # Update weights and biases
            self.w3 -= lr * np.outer(h2, delta3)
            self.b3 -= lr * delta3
            self.w2 -= lr * np.outer(h1, delta2)
            self.b2 -= lr * delta2
            self.w1 -= lr * np.outer(state, delta1)
            self.b1 -= lr * delta1
    
    def update_target_model(self):
        """Update target model to match current model."""
        if TORCH_AVAILABLE:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            # Copy weights and biases to target model
            self.target_w1 = self.w1.copy()
            self.target_b1 = self.b1.copy()
            self.target_w2 = self.w2.copy()
            self.target_b2 = self.b2.copy()
            self.target_w3 = self.w3.copy()
            self.target_b3 = self.b3.copy()
    
    def save_model(self, path):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, path)
        else:
            # Save NumPy model
            np.savez(
                path,
                w1=self.w1, b1=self.b1,
                w2=self.w2, b2=self.b2,
                w3=self.w3, b3=self.b3
            )
    
    def load_model(self, path):
        """Load the model from disk."""
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False
        
        try:
            if TORCH_AVAILABLE:
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.update_target_model()
            else:
                # Load NumPy model
                weights = np.load(path)
                self.w1 = weights['w1']
                self.b1 = weights['b1']
                self.w2 = weights['w2']
                self.b2 = weights['b2']
                self.w3 = weights['w3']
                self.b3 = weights['b3']
                self.update_target_model()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class ReplayMemory:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from memory."""
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self):
        return len(self.memory)


class ValetRLAgent:
    """Reinforcement learning agent for Valet Parking Game."""
    
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = ACTION_SPACE
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = DQNModel(state_size, self.action_size)
        self.step_count = 0
        
        # Performance metrics
        self.total_reward = 0
        self.episode_count = 0
        self.stats = {
            'episodes': [],
            'rewards': [],
            'epsilon': [],
            'successful_parks': 0,
            'successful_deliveries': 0,
        }
        
        # Game state tracking
        self.current_car = None
        self.target_parking_spot = None
        self.client_car_map = {}  # Maps clients to their cars
        self.car_client_map = {}  # Maps cars to their clients
        self.parked_cars = {}     # Maps parked cars to their parking spots
        
        # Path planning and state representation
        self.known_parking_spots = {}  # Mapping of parking spot numbers to locations
    
    def get_state(self, player, cars, clients, spot_rectangles):
        """
        Extract and normalize state from game entities.
        
        Args:
            player: The player sprite
            cars: List of car sprites
            clients: List of client sprites
            spot_rectangles: List of parking spot rectangles
        
        Returns:
            numpy array representing the state
        """
        # Store parking spot locations
        for spot in spot_rectangles:
            spot_num = spot["number"]
            spot_rect = spot["rect"]
            self.known_parking_spots[spot_num] = (spot_rect.centerx, spot_rect.centery)
        
        # Normalize coordinates to 0-1 range
        screen_width = 1366  # Game window width
        screen_height = 768  # Game window height
        
        # Start with player position (or -1,-1 if player is in a car)
        state = []
        if player.active:
            state.extend([
                player.rect.centerx / screen_width,
                player.rect.centery / screen_height
            ])
        else:
            state.extend([-1, -1])  # Player is in a car
        
        # Add information for each car (up to a fixed number)
        max_cars = 10  # Maximum number of cars to track in state
        car_info = []
        
        for i, car in enumerate(cars):
            if i >= max_cars:
                break
                
            # Convert ParkingSpot to integer before division
            try:
                parking_spot_normalized = float(car.ParkingSpot) / 2000.0
            except (ValueError, TypeError):
                # If conversion fails, use a default value
                print(f"Warning: Could not convert ParkingSpot '{car.ParkingSpot}' to float. Using default value.")
                parking_spot_normalized = 0.5  # Use middle value as default
                
            # Basic car properties
            car_info.extend([
                car.rect.centerx / screen_width,
                car.rect.centery / screen_height,
                car.angle / 360.0,  # Normalized angle
                1.0 if car.active else 0.0,  # Is this the active car?
                1.0 if car.parked else 0.0,  # Is the car parked?
                parking_spot_normalized,  # Normalized parking spot number
                1.0 if car.ClientEntered else 0.0,  # Has the client entered?
                1.0 if car.Client.sprite.ClientExited else 0.0,  # Has client exited?
                car.parking_alignment / 100.0  # Parking alignment
            ])
            
            # Client position for this car
            if car.Client.sprite:
                car_info.extend([
                    car.Client.sprite.ClientX / screen_width,
                    car.Client.sprite.ClientY / screen_height
                ])
            else:
                car_info.extend([0, 0])
        
        # Pad with zeros if we have fewer than max_cars
        while len(car_info) < max_cars * 11:  # 9 car props + 2 client props
            car_info.extend([0] * 11)
            
        state.extend(car_info)
        
        return np.array(state, dtype=np.float32)
    
    def act(self, state, training=True):
        """
        Choose an action based on state using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether we're in training mode (use epsilon-greedy)
            
        Returns:
            Action index (0-5)
        """
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.forward(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def step(self, state, action, reward, next_state, done):
        """
        Process a step in the environment.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Update total reward
        self.total_reward += reward
        
        # Periodically train the model
        if len(self.memory) > TRAINING_START and len(self.memory) > BATCH_SIZE:
            self._train_model()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % UPDATE_TARGET_FREQUENCY == 0:
            self.model.update_target_model()
        
        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def _train_model(self):
        """Train the model using a batch from the replay buffer."""
        minibatch = self.memory.sample(BATCH_SIZE)
        
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        if TORCH_AVAILABLE:
            # PyTorch training
            # Get max Q value for next states from target model
            next_q_values = self.model.target_forward(next_states)
            next_q_values_max = np.max(next_q_values, axis=1)
            
            # Calculate target Q values
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values_max
            
            # Get current Q values and update with target
            current_q_values = self.model.forward(states)
            targets = current_q_values.copy()
            
            for i in range(BATCH_SIZE):
                targets[i, actions[i]] = target_q_values[i]
            
            # Train the model
            self.model.train(states, targets)
        else:
            # NumPy training (simplified)
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                # Get predicted Q-values for all actions
                target = self.model.forward(state.reshape(1, -1))[0]
                
                # If done, use only reward
                if done:
                    target[action] = reward
                else:
                    # Use Bellman equation: Q(s,a) = r + γ * max(Q(s',a'))
                    next_q_values = self.model.target_forward(next_state.reshape(1, -1))[0]
                    target[action] = reward + GAMMA * np.max(next_q_values)
                
                # Train model with this single sample
                self.model.train(state.reshape(1, -1), target.reshape(1, -1))
    
    def end_episode(self, episode, successful_parks=0, successful_deliveries=0):
        """Log stats at the end of an episode."""
        self.episode_count += 1
        self.stats['episodes'].append(episode)
        self.stats['rewards'].append(self.total_reward)
        self.stats['epsilon'].append(self.epsilon)
        self.stats['successful_parks'] += successful_parks
        self.stats['successful_deliveries'] += successful_deliveries
        
        # Reset episode stats
        self.total_reward = 0
        
        if episode % 10 == 0:
            avg_reward = np.mean(self.stats['rewards'][-10:])
            print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
    
    def save(self, checkpoint_dir='checkpoints'):
        """Save agent state and model."""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, 'dqn_model.pth') if TORCH_AVAILABLE else os.path.join(checkpoint_dir, 'dqn_model.npz')
        self.model.save_model(model_path)
        
        # Save agent stats
        stats_path = os.path.join(checkpoint_dir, 'agent_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f)
        
        print(f"Agent saved to {checkpoint_dir}")
    
    def load(self, checkpoint_dir='checkpoints'):
        """Load agent state and model."""
        # Load model
        model_path = os.path.join(checkpoint_dir, 'dqn_model.pth') if TORCH_AVAILABLE else os.path.join(checkpoint_dir, 'dqn_model.npz')
        if self.model.load_model(model_path):
            # Load agent stats if model loaded successfully
            stats_path = os.path.join(checkpoint_dir, 'agent_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
                print(f"Agent loaded from {checkpoint_dir}")
                return True
        return False


class ActionMapper:
    """Maps RL agent's actions to game inputs."""
    
    @staticmethod
    def get_action_keys(action):
        """
        Convert agent action to keyboard inputs.
        
        Args:
            action: Action index from agent (0-5)
            
        Returns:
            Dictionary with key states for pygame
        """
        # Create an empty dictionary for key states
        keys = {}
        keys[pygame.K_UP] = action == 0
        keys[pygame.K_DOWN] = action == 1
        keys[pygame.K_LEFT] = action == 2
        keys[pygame.K_RIGHT] = action == 3
        keys[pygame.K_SPACE] = action == 4
        # Action 5 is do nothing, so all keys are False
        
        return keys
    
    @staticmethod
    def simulate_key_press(key_code):
        """
        Simulate a key press event.
        
        Args:
            key_code: Pygame key code
            
        Returns:
            Pygame key event
        """
        return pygame.event.Event(pygame.KEYDOWN, {"key": key_code})
    
    @staticmethod
    def simulate_key_release(key_code):
        """
        Simulate a key release event.
        
        Args:
            key_code: Pygame key code
            
        Returns:
            Pygame key event
        """
        return pygame.event.Event(pygame.KEYUP, {"key": key_code})


def calculate_reward(
    car_sprite, 
    previous_state, 
    current_state,
    action_taken,
    spot_rectangles,
    collision_occurred=False
):
    """
    Calculate reward based on current game state.
    
    Args:
        car_sprite: The active car sprite or None
        previous_state: Previous state array
        current_state: Current state array
        action_taken: Action index that was taken
        spot_rectangles: List of parking spot rectangles
        collision_occurred: Whether a collision occurred
        
    Returns:
        Calculated reward value
    """
    reward = 0
    
    # Penalty for collisions
    if collision_occurred:
        reward -= 10.0
    
    # Check for successful park
    if car_sprite and not car_sprite.parked and current_state[5]:  # Car newly parked
        # Check if parked in correct spot
        if car_sprite.current_spot == car_sprite.assigned_spot:
            reward += 20.0
        else:
            # Wrong spot penalty
            reward += 5.0  # Still give some reward for parking, but less
    
    # Check for successful delivery
    if car_sprite and car_sprite.SuccessDelivery:
        reward += 30.0
    
    # Small time penalty to encourage efficiency
    reward -= 0.1
    
    # Movement towards goal reward
    if car_sprite:
        # If we have a target spot, reward movement toward it
        if car_sprite.assigned_spot and not car_sprite.parked:
            # Find the target spot
            target_spot = None
            for spot in spot_rectangles:
                if spot["number"] == car_sprite.assigned_spot:
                    target_spot = spot["rect"]
                    break
            
            if target_spot:
                # Calculate distances
                prev_distance = np.sqrt((car_sprite.last_position.x - target_spot.centerx)**2 + 
                                       (car_sprite.last_position.y - target_spot.centery)**2)
                current_distance = np.sqrt((car_sprite.rect.centerx - target_spot.centerx)**2 + 
                                          (car_sprite.rect.centery - target_spot.centery)**2)
                
                # Reward for getting closer to target
                distance_change = prev_distance - current_distance
                reward += distance_change * 0.5
                
                # Reward for better parking alignment
                if car_sprite.parking_alignment > 0:
                    reward += (car_sprite.parking_alignment / 100.0)
    
    return reward