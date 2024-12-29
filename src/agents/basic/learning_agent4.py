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
            random_letter = random.choice(self.alphabet)
            self.state = self.name.find(random_letter)
            if self.state == -1:
                self.state = 0
            print(f"Starting at letter: {random_letter}")
        else:
            self.state = 0
            print(f"Starting at letter: {self.name[self.state]}")
        return self.state

    def step(self, action):
        if action == 0:
            self.state = min(self.state + 1, len(self.name) - 1)
        elif action == 1:
            self.state = max(self.state - 1, 0)

        if self.state == len(self.name) - 1:
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False

        return self.state, reward, done

def initialize_q_table(file_path, name_length):
    if os.path.exists(file_path):
        q_table = np.load(file_path)
        print("Q-table loaded from file.")
    else:
        q_table = np.zeros((name_length, 2))
        print("Q-table initialized.")
    return q_table

def choose_action(state, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1])
    else:
        return np.argmax(q_table[state])

def train_agent(env, q_table, alpha, gamma, epsilon, num_episodes, file_path):
    for episode in range(num_episodes):
        state = env.reset(training=True)
        done = False
        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done = env.step(action)
            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )
            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}: Q-table {q_table}")

    np.save(file_path, q_table)
    print("Q-table saved to file.")

def test_agent(env, q_table):
    # Identify the first letter using the Q-table
    first_letter_index = np.argmax(q_table[:, 0] + q_table[:, 1])
    print(f"Identified first letter: {env.name[first_letter_index]}")

    state = first_letter_index
    done = False
    current_guess = [env.name[state]] if state != -1 else []

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        if state != -1:
            current_guess.append(env.name[state])
        print(f"Current Guess: {''.join(current_guess)}, State: {state}, Action: {chr(action + ord('A'))}, Reward: {reward}")

    print(f"Final Guess: {''.join(current_guess)}")

def main():
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000
    q_table_file = 'q_table_name4.npy'
    NAME_TO_LEARN = "vic"  # Change this to any name or sequence you want to learn

    env = NameLearningEnvironment(NAME_TO_LEARN)
    q_table = initialize_q_table(q_table_file, len(env.name))

    if not os.path.exists(q_table_file):
        train_agent(env, q_table, alpha, gamma, epsilon, num_episodes, q_table_file)

    for i in range(5):
        print(f"\nTest {i + 1}:")
        test_agent(env, q_table)

if __name__ == "__main__":
    main()