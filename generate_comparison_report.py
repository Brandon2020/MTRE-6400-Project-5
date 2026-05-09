"""
Generates comparison table and trajectory plots for TD3 vs SAC+HER
"""

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3, SAC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

gym.register_envs(gymnasium_robotics)
env = gym.make("AntMaze_Medium-v5")

# Load models
td3_model = TD3.load("models/td3_saved_on_interrupt.zip", env=env)
sac_model = SAC.load("models/best_model.zip" if os.path.exists("models/best_model.zip") else "models/sac_her_final.zip", env=env)

def evaluate_model(model, n_episodes=30, name=""):
    print(f"Evaluating {name}...")
    returns, lengths, successes, distances = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ret = length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            length += 1
            done = terminated or truncated
        success = info.get("success", False) or ret > 0
        dist = np.linalg.norm(obs['observation'][:2] - obs['desired_goal'][:2])
        returns.append(ret)
        lengths.append(length)
        successes.append(success)
        distances.append(dist)
    return {
        "Algorithm": name,
        "Success Rate (%)": round(100 * np.mean(successes), 1),
        "Avg Return": round(np.mean(returns), 2),
        "Avg Episode Length": round(np.mean(lengths), 1),
        "Avg Final Goal Distance": round(np.mean(distances), 3),
    }

td3_res = evaluate_model(td3_model, 30, "TD3")
sac_res = evaluate_model(sac_model, 30, "SAC+HER")

df = pd.DataFrame([td3_res, sac_res])
print(df.to_string(index=False))

# Save table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
plt.title("TD3 vs SAC+HER Performance Comparison")
plt.savefig("comparison_table.png", dpi=300, bbox_inches='tight')
print("✅ comparison_table.png saved")
