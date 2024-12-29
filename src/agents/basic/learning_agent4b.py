import numpy as np
import os
import random
import string

class NameLearningEnvironment:
    def __init__(self, name):
        self.name = name
        self.state = 0
        self.alphabet = string.ascii_lowercase

    def reset(self, training=True):
        if training:
            self.state = 0  # Always start at the first letter during training
            print(f"Starting at letter: {self.name[self.state]}")
        else:
            self.state = 0  # During testing, we begin from the first letter
            print(f"Starting at letter: {self.name[self.state]}")
        return self.state

    def step(self, action):
        if action == 0:
            self.state = min(self.state + 1, len(self.name) - 1)
        elif action == 1:
            self.state = max(self.state - 1, 0)

        # If the agent is at the last character, mark as done and give a reward
        if self.state == len(self.name) - 1:
            reward = 1
            done = True
        else:
            reward = -0.1  # Continue training with a small penalty
            done = False

        return self.state, reward, done

def initialize_q_table(file_path, name_length):
    if os.path.exists(file_path):
        q_table = np.load(file_path)
        print("Q-table loaded from file.")
    else:
        q_table = np.zeros((name_length, 2))  # Two actions: move forward (0) or backward (1)
        print("Q-table initialized.")
    return q_table

def choose_action(state, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1])  # Random action (exploration)
    else:
        return np.argmax(q_table[state])  # Best action (exploitation)

def train_agent(env, q_table, alpha, gamma, epsilon, num_episodes, file_path):
    for episode in range(num_episodes):
        state = env.reset(training=True)
        done = False
        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done = env.step(action)
            # Update Q-table using the Q-learning formula
            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )
            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}: Q-table {q_table}")

    np.save(file_path, q_table)
    print("Q-table saved to file.")

def test_agent(env, q_table):
    state = env.reset(training=False)
    done = False
    current_guess = [env.name[state]]  # Start with the first letter in the name

    while not done:
        action = np.argmax(q_table[state])  # Choose best action based on learned policy
        next_state, reward, done = env.step(action)
        current_guess.append(env.name[next_state])  # Append the guessed letter
        print(f"Current Guess: {''.join(current_guess)}, State: {state}, Reward: {reward}")

    print(f"Final Guess: {''.join(current_guess)}")

def main():
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000
    q_table_file = 'q_table_name4b.npy'
    NAME_TO_LEARN = "blues-and-rhythm"  # Change this to any name or sequence you want to learn

    env = NameLearningEnvironment(NAME_TO_LEARN)
    q_table = initialize_q_table(q_table_file, len(env.name))

    if not os.path.exists(q_table_file):
        train_agent(env, q_table, alpha, gamma, epsilon, num_episodes, q_table_file)

    for i in range(5):
        print(f"\nTest {i + 1}:")
        test_agent(env, q_table)

if __name__ == "__main__":
    main()
