# Meta-Learning-Integration-in-Hierarchical-Reinforcement-Learning-for-Advanced-Task-Complexity
This repository hosts the implementation and resources for the research project titled "Meta-Learning Integration in Hierarchical Reinforcement Learning for Advanced Task Complexity." 

## Scenarios
### Scenario 1: Fixed Complexity Environment
In this scenario, the agent operates in a grid environment of fixed size (6x6) with 3 traps. The task is for the agent to navigate the grid from the starting point to a goal while avoiding traps. The policy network is trained using meta-learning techniques, focusing on a fixed level of complexity.

### Key Parts of the Code
### Custom Grid Environment
- A 6x6 grid is initialized, with 3 traps randomly placed across the grid.
- The agent starts at the top-left corner, and the goal is at the bottom-right corner.
- The step function defines how the agent moves (up, down, left, right) and calculates rewards for reaching the goal, falling into traps, or simply moving.

```python
class CustomGridEnv:
def __init__(self, size=6, num_traps=3):
    self.size = size
    self.num_traps = num_traps
    # Initialize grid, traps, agent position, etc.
def step(self, action):
    # Define agent movement and rewards
    pass
```

### Policy Network
- A neural network represents the policy.
- The input is a one-hot encoded state, and the output is the Q-values for the four possible actions (up, down, left, right).

```python
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, input_size, num_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_actions)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```


### State Representation and Action Selection
- The state is converted into a one-hot encoded vector, which is then passed through the policy network.
- Epsilon-greedy action selection is used to balance exploration and exploitation.

```python
def select_action(network, state, epsilon, num_actions):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)
    else:
        with torch.no_grad():
            q_values = network(state)
            return torch.argmax(q_values).item()
```

### Meta-Training Process
- The agent is trained for a fixed number of iterations in the grid environment, with a focus on optimizing the policy network using meta-learning techniques.
- The learning rate is gradually reduced, and the epsilon value for exploration decays over time.

```python
def meta_train_fixed_complexity(meta_learning_rate, epsilon_start, epsilon_decay, num_iterations, num_inner_steps):
    # Initialize environment, network, optimizer, etc.
    for iteration in range(num_iterations):
        # Meta-training loop
        pass
```


## Scenario 2: Curriculum Learning with Variable Complexity
This scenario introduces curriculum learning, where the agent is progressively exposed to environments with increasing complexity. The grid sizes and the number of traps increase as the agent becomes more proficient, allowing it to generalize across different levels of difficulty.

### Key Parts of the Code
- Custom Grid Environment with Variable Complexity
- The grid size and number of traps vary with each level of the curriculum.

```python
class CustomGridEnv:
    def __init__(self, size=4, num_traps=1):
        self.size = size
        self.num_traps = num_traps
        # Initialize grid, traps, agent position, etc.
```


### Curriculum Levels
- The curriculum consists of multiple levels, each with a different grid size and number of traps.



```python
curriculum_levels = [
    {'level': 1, 'size': 4, 'num_traps': 1, 'num_tasks': 20, 'iterations': 1000},
    {'level': 2, 'size': 6, 'num_traps': 3, 'num_tasks': 20, 'iterations': 1500},
    {'level': 3, 'size': 8, 'num_traps': 4, 'num_tasks': 20, 'iterations': 2000}
]
```
### Meta-Training with Curriculum Learning
- The agent is trained across multiple levels.
- In each level, it faces a set of tasks where the complexity increases after achieving success in the current tasks.
- The same policy network from Scenario 1 is used but now adapts to environments of increasing difficulty.

```python
def meta_train_with_complexities(meta_learning_rate, epsilon_start, epsilon_decay, num_inner_steps):
    for level in curriculum_levels:
        # Training loop for each level
        pass
```
### State and Action Updates
As in Scenario 1, the state is one-hot encoded, and actions are selected using an epsilon-greedy strategy.

```python
def select_action(network, state, epsilon, num_actions):
    # Same as in Scenario 1
    pass
```

## Results
Include graphs, metrics, or any visualizations that demonstrate the performance improvements achieved through your approach.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this code, provided that proper attribution is given to the original authors.

## Citation

### Paper Information
**Paper Title:** Meta-Learning Integration in Hierarchical Reinforcement Learning for Advanced Task Complexity
**Authors:** Arash Khajooeinejad, Masoumeh Chapariniya
**Link:** [Meta-Learning Integration in Hierarchical Reinforcement Learning for Advanced Task Complexity](https://arxiv.org/abs/2410.07921)


If you use this code in your research, please cite the corresponding paper and the repository:
```shell
@article{Khajooeinejad2024,
  title={Meta-Learning Integration in Hierarchical Reinforcement Learning for Advanced Task Complexity},
  author={Arash Khajooeinejad and Masoumeh Chapariniya},
  journal={arXiv preprint arXiv:2410.07921},
  year={2024},
  howpublished={\url{https://arxiv.org/abs/2410.07921}},
}
```
