# data_loaders/cartpole.py

import gym

def load_cartpole():
    env = gym.make('CartPole-v1')
    return env

# Load the environment and run a simple test
if __name__ == "__main__":
    env = load_cartpole()
    env.reset()
    for _ in range(1000):
        # print("Here")
        env.render()
        action = env.action_space.sample()  # take a random action
        env.step(action)
    env.close()