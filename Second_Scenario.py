# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for DataFrame handling

# Suppress possible warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define the device for computation (use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom grid environment with variable sizes and traps
class CustomGridEnv:
    def __init__(self, size=4, num_traps=1):
        self.size = size
        self.num_traps = num_traps
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self._generate_traps()
        return self._get_obs()

    def _generate_traps(self):
        self.traps = []
        while len(self.traps) < self.num_traps:
            trap = [np.random.randint(self.size), np.random.randint(self.size)]
            if trap != self.agent_pos and trap != self.goal_pos:
                self.traps.append(trap)

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:  # Down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:  # Right
            self.agent_pos[1] += 1

        done = False
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        elif self.agent_pos in self.traps:
            reward = -5
            done = True
        else:
            reward = -1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

# Neural network for the policy
class PolicyNet(nn.Module):
    def __init__(self, input_size, num_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

# Helper function to convert a state to one-hot representation
def state_to_one_hot(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return torch.FloatTensor([one_hot]).to(device)

# Epsilon-greedy action selection
def select_action(network, state, epsilon, num_actions):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)
    else:
        with torch.no_grad():
            q_values = network(state)
            return torch.argmax(q_values).item()

# Curriculum levels with increasing complexity
curriculum_levels = [
    {'level': 1, 'size': 4, 'num_traps': 1, 'num_tasks': 20, 'iterations': 1000},  # Easier level
    {'level': 2, 'size': 6, 'num_traps': 3, 'num_tasks': 20, 'iterations': 1500},  # Medium complexity
    {'level': 3, 'size': 8, 'num_traps': 4, 'num_tasks': 20, 'iterations': 2000}  # Harder level
]

# Meta-training process with gradual complexity adaptation
def meta_train_with_complexities(meta_learning_rate, epsilon_start, epsilon_decay, num_inner_steps):
    num_states = 8 * 8  # Assume the largest grid size (8x8)
    num_actions = 4
    discount_factor = 0.99

    # Initialize policy network
    policy_net = PolicyNet(input_size=num_states, num_actions=num_actions).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=meta_learning_rate)

    epsilon = epsilon_start
    meta_losses, meta_rewards, success_rates = [], [], []

    for level_info in curriculum_levels:
        env = CustomGridEnv(size=level_info['size'], num_traps=level_info['num_traps'])
        print(f"Starting Level {level_info['level']} with {level_info['iterations']} iterations")

        for iteration in range(level_info['iterations']):
            total_loss = 0
            total_reward = 0
            successes = 0

            for task in range(level_info['num_tasks']):
                state = env.reset()
                state = state_to_one_hot(state, num_states)
                optimizer.zero_grad()

                for step in range(num_inner_steps):
                    action = select_action(policy_net, state, epsilon, num_actions)
                    next_state, reward, done, _ = env.step(action)
                    next_state = state_to_one_hot(next_state, num_states)

                    with torch.no_grad():
                        target = reward + discount_factor * torch.max(policy_net(next_state))
                    prediction = policy_net(state)[0][action]
                    loss = nn.functional.smooth_l1_loss(prediction, target)
                    loss.backward()
                    total_loss += loss.item()

                    optimizer.step()
                    state = next_state
                    total_reward += reward
                    if done:
                        if reward == 10:  # Success is defined as reaching the goal
                            successes += 1
                        break

            # Store results with moving average to smooth out success rates
            meta_losses.append(total_loss / level_info['num_tasks'])
            meta_rewards.append(total_reward / level_info['num_tasks'])
            success_rates.append(successes / level_info['num_tasks'])
            epsilon = max(0.1, epsilon * epsilon_decay)

    return meta_losses, meta_rewards, success_rates

# Function to calculate moving average
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Plot function for meta-loss, average reward, and success rate
def plot_meta_losses_rewards_success(meta_losses, meta_rewards, success_rates, window_size=30):
    smoothed_losses = moving_average(meta_losses, window_size)
    smoothed_rewards = moving_average(meta_rewards, window_size)
    smoothed_success_rates = moving_average(success_rates, window_size)

    # Create the figure and axes with white background
    fig, ax1 = plt.subplots(figsize=(14, 7), facecolor='white')

    # Set axes background color to white
    ax1.set_facecolor('white')

    color = 'tab:red'
    ax1.set_xlabel('Meta-Iteration')
    ax1.set_ylabel('Meta-Loss', color=color)
    ax1.plot(meta_losses, color=color, alpha=0.1, label='Meta-Loss', marker='o', markersize=2)
    ax1.plot(range(window_size - 1, len(meta_losses)), smoothed_losses, color=color, label=f'Smoothed Meta-Loss (window={window_size})', marker='o', markersize=2)

    # Add vertical dashed lines and label them
    ax1.axvline(x=1000, color='gray', linestyle='--', label='Level Transition 1')
    ax1.axvline(x=2500, color='gray', linestyle='--', label='Level Transition 2')
    ax1.axvline(x=4500, color='gray', linestyle='--', label='Level Transition 3')

    ax1.tick_params(axis='y', labelcolor=color)

    # Twin x-axis for Average Reward
    ax2 = ax1.twinx()
    ax2.set_facecolor('white')  # Set the background color of the second axis to white
    color = 'tab:blue'
    ax2.set_ylabel('Average Reward', color=color)
    ax2.plot(meta_rewards, color=color, alpha=0.1, label='Average Reward', marker='s', markersize=2)
    ax2.plot(range(window_size - 1, len(meta_rewards)), smoothed_rewards, color=color, label=f'Smoothed Average Reward (window={window_size})', marker='s', markersize=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Third axis for Success Rate
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outwards
    ax3.set_facecolor('white')  # Set the background color of the third axis to white
    color = 'tab:green'
    ax3.set_ylabel('Success Rate', color=color)
    ax3.plot(success_rates, color=color, alpha=0.1, label='Success Rate', marker='^', markersize=2)
    ax3.plot(range(window_size - 1, len(success_rates)), smoothed_success_rates, color=color, label=f'Smoothed Success Rate (window={window_size})', marker='^', markersize=2)
    ax3.tick_params(axis='y', labelcolor=color)

    # Add a legend to the plot
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='lower right')

    plt.title("Meta-Loss, Average Reward, and Success Rate Progress")
    fig.tight_layout()  # Adjust layout to prevent label clipping
    plt.grid(True)
    plt.show()


# Simplified run with multiple complexity levels and success rate tracking
if __name__ == "__main__":
    meta_learning_rate = 1e-3
    epsilon_start = 0.9
    epsilon_decay = 0.99
    num_inner_steps = 50  # Number of steps the agent will take within each task

    # Run the meta-training process with multiple complexities
    meta_losses, meta_rewards, success_rates = meta_train_with_complexities(
        meta_learning_rate=meta_learning_rate,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        num_inner_steps=num_inner_steps
    )

    # Plot results
    plot_meta_losses_rewards_success(meta_losses, meta_rewards, success_rates)
