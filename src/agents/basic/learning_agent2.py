import numpy as np

class NameLearningEnvironment:
    def __init__(self):
        # Define the name to be learned
        self.name = "vic"
        self.state = 0  # Initial state (first letter of "vic")

    def reset(self):
        self.state = 0  # Start at the first letter ("v")
        print(f"Starting at letter: {self.name[self.state]}")  # Print where it starts
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

# Function to test if the agent remembers how to spell "vic"
def test_agent():
    state = name_env.reset()
    done = False
    current_guess = []  # List to store the agent's guesses

    while not done:
        action = np.argmax(q_table_name[state])  # Choose the best action based on the Q-table
        state, reward, done = name_env.step(action)

        # Append the current letter to the guess list
        current_guess.append("vic"[state])  # Use the state to index the name

        # Print the current guess
        print(f"Current Guess: {''.join(current_guess)}, State: {state}, Action: {chr(action + ord('A'))}, Reward: {reward}")

    # Final output to check if the name was learned correctly
    print(f"Final Guess: {''.join(current_guess)}")

# Test the agent multiple times
for i in range(5):
    print(f"\nTest {i + 1}:")
    test_agent()