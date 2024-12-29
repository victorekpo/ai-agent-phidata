import numpy as np

class NameLearningEnvironment:
    def __init__(self):
        # Define the name to be learned
        self.name = "vic"
        self.state = 0  # Initial state (first letter of "vic")

    def reset(self):
        self.state = 0  # Start at the first letter
        return self.state

    def step(self, action):
        # Transition from current state based on the action
        if action == 0:  # Action 0: Move to next letter
            self.state = min(self.state + 1, len(self.name) - 1)
        elif action == 1:  # Action 1: Move to previous letter
            self.state = max(self.state - 1, 0)

        # Define rewards
        if self.state == len(self.name) - 1:  # Reached the last letter
            reward = 1  # Goal state (learning "vic" complete)
            done = True
        else:
            reward = -0.1  # Penalty for each step
            done = False

        return self.state, reward, done

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

# Initialize Q-table for NameLearningEnvironment
q_table_name = np.zeros((len("vic"), 2))  # 3 states (for "v", "i", "c") and 2 actions (move forward/backward)

def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1])  # Explore
    else:
        return np.argmax(q_table_name[state])  # Exploit

name_env = NameLearningEnvironment()

# Training phase for NameLearningEnvironment
for episode in range(num_episodes):
    state = name_env.reset()
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = name_env.step(action)

        # Update Q-value
        q_table_name[state][action] = q_table_name[state][action] + alpha * (
            reward + gamma * np.max(q_table_name[next_state]) - q_table_name[state][action]
        )

        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode}: Q-table {q_table_name}")

# Testing phase for NameLearningEnvironment
state = name_env.reset()
done = False
while not done:
    action = np.argmax(q_table_name[state])
    state, reward, done = name_env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")
