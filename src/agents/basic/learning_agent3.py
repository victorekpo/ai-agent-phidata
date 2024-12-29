import numpy as np
import os

class NameLearningEnvironment:
    def __init__(self):
        self.name = "vic"
        self.state = 0

    def reset(self):
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
        state = env.reset()
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
    state = env.reset()
    done = False
    current_guess = []

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        current_guess.append("vic"[state])
        print(f"Current Guess: {''.join(current_guess)}, State: {state}, Action: {chr(action + ord('A'))}, Reward: {reward}")

    print(f"Final Guess: {''.join(current_guess)}")

def main():
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 1000
    q_table_file = 'q_table_name.npy'

    env = NameLearningEnvironment()
    q_table = initialize_q_table(q_table_file, len(env.name))

    if not os.path.exists(q_table_file):
        train_agent(env, q_table, alpha, gamma, epsilon, num_episodes, q_table_file)

    for i in range(5):
        print(f"\nTest {i + 1}:")
        test_agent(env, q_table)

if __name__ == "__main__":
    main()