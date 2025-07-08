# src/agent/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([s[0] for s in minibatch])).to(self.device)
        actions = torch.LongTensor([s[1] for s in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in minibatch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([s[3] for s in minibatch])).to(self.device)
        dones = torch.FloatTensor([s[4] for s in minibatch]).unsqueeze(1).to(self.device)
        
        # Get Q-values for current states from the policy net
        q_values = self.policy_net(states).gather(1, actions)
        
        # Get next Q-values from the target net
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Compute the expected Q-values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())