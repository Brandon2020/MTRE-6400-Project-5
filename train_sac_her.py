"""
SAC + HER Training Script for AntMaze_Medium-v5
Well-commented version for your project report and code submission.
"""

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import os

# ==================== 1. Register Environments ====================
# Gymnasium-Robotics environments need to be registered manually
gym.register_envs(gymnasium_robotics)

# ==================== 2. Create Folders ====================
# Store logs for TensorBoard and trained models
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==================== 3. Create the Environment ====================
# Using Medium maze as requested (not BigMaze)
env = gym.make("AntMaze_Medium-v5")

# ==================== 4. SAC + HER Setup ====================
# SAC = Soft Actor-Critic (Maximum Entropy RL)
# HER = Hindsight Experience Replay (crucial for sparse reward mazes)

model = SAC(
    "MultiInputPolicy",           # Required for goal-conditioned observations (observation + desired_goal)
    env,
    verbose=1,                    # Print training progress
    tensorboard_log="./logs/sac_her/",   # For live learning curves
    learning_starts=10000,        # Wait longer before training (HER needs full episodes)
    buffer_size=1_000_000,        # Large buffer for HER
    batch_size=256,
    gamma=0.99,                   # Discount factor
    tau=0.005,                    # Soft target update
    learning_rate=1e-3,
    
    # ==================== HER Configuration ====================
    replay_buffer_class=HerReplayBuffer,   # Enable Hindsight Experience Replay
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                     # Sample 4 virtual goals per real transition
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,  # Use future states as goals
    ),
)

# ==================== 5. Evaluation Callback ====================
# Automatically evaluates the agent and saves the best model
eval_callback = EvalCallback(
    gym.make("AntMaze_Medium-v5"),
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,              # Evaluate every 10k steps
    n_eval_episodes=10,
    deterministic=True,           # No exploration noise during evaluation
)

# ==================== 6. Start Training ====================
print("🚀 Starting SAC + HER training on AntMaze_Medium-v5...")
print("   SAC = Soft Actor-Critic (maximum entropy)")
print("   HER = Hindsight Experience Replay (learns from failures)")
print("   Press Ctrl + C to stop training gracefully.")

model.learn(
    total_timesteps=1_000_000,    # You can increase this
    callback=eval_callback
)

# ==================== 7. Save Final Model ====================
model.save("models/sac_her_final")
print("\n✅ Training finished!")
print("   Final model saved as: models/sac_her_final.zip")
print("   Best model saved as: models/best_model.zip")
