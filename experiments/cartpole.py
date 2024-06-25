import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.callbacks import ProgressBarCallback
import torch
from torch.optim import Adam

# Custom optimizer
class CustomAdam(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(CustomAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# Custom DQN class
class CustomDQN(DQN):
    def _setup_model(self):
        super()._setup_model()
        # Replace the default optimizer with the custom optimizer
        self.policy.optimizer = CustomAdam(self.policy.parameters(), lr=1e-3)

# Set up the environment and wrap it in DummyVecEnv
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# Instantiate the custom DQN model
model = CustomDQN(MlpPolicy, env, verbose=1)

# Train the model with a progress bar
total_timesteps = 10000  # Adjust as needed
progress_callback = ProgressBarCallback()
model.learn(total_timesteps=total_timesteps, callback=progress_callback)

# Save the model
model.save("custom_dqn_cartpole")
print("Model saved as custom_dqn_cartpole")

# Load the model (optional, for evaluation purposes)
model = CustomDQN.load("custom_dqn_cartpole", env=env)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    env.render()
    if dones[0]:  # For DummyVecEnv, dones is a list
        obs = env.reset()

env.close()
