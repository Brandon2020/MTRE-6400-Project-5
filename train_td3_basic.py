"""
Basic TD3 Training Script for AntMaze_Medium-v5
This is a clean, well-commented version for your project.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import os

# ==================== 1. Register Environments ====================
# Gymnasium-Robotics environments are not registered by default in newer versions.
# We must register them manually so "AntMaze_Medium-v5" can be created.
gym.register_envs(gymnasium_robotics)

# ==================== 2. Create Folders ====================
# We need folders to save logs (for TensorBoard) and trained models.
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==================== 3. Create the Environment ====================
# AntMaze_Medium-v5 is a goal-conditioned environment with sparse rewards.
# The agent (Ant) must learn to reach the goal in a maze.
env = gym.make("AntMaze_Medium-v5")

# ==================== 4. TD3 Setup ====================
# TD3 = Twin Delayed Deep Deterministic Policy Gradient
# Good for continuous control tasks like robot locomotion.

# Action noise helps with exploration (very important in sparse reward mazes)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), 
                                 sigma=0.1 * np.ones(n_actions))

model = TD3(
    "MultiInputPolicy",           # Important: Use MultiInputPolicy because observation has 'observation' + 'desired_goal'
    env,
    action_noise=action_noise,    # Helps exploration
    verbose=1,                    # Print training info
    tensorboard_log="./logs/td3_basic/",   # For live graphs
    learning_starts=5000,         # Wait before training to fill replay buffer
    buffer_size=500_000,          # Large replay buffer for stability
    batch_size=256,
    gamma=0.99,                   # Discount factor
    tau=0.005,                    # Soft update for target networks
    policy_delay=2,               # Delay actor updates (core TD3 idea)
    learning_rate=1e-3,
)

# ==================== 5. Evaluation Callback ====================
# This automatically evaluates the agent every 10k steps and saves the best model.
eval_callback = EvalCallback(
    gym.make("AntMaze_Medium-v5"),      # Separate eval environment
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,                    # Evaluate every 10k steps
    n_eval_episodes=10,                 # Average over 10 episodes
    deterministic=True,                 # No noise during evaluation
)

# ==================== 6. Start Training ====================
print("🚀 Starting BASIC TD3 training on AntMaze_Medium-v5...")
print("   (Press Ctrl + C to stop gracefully when you want)")

model.learn(
    total_timesteps=500_000,      # You can increase this (e.g. 1_000_000)
    callback=eval_callback
)

# ==================== 7. Save Final Model ====================
model.save("models/td3_basic_final")
print("✅ Training finished! Model saved in models/ folder")
print("   Best model is saved as 'best_model.zip'")
