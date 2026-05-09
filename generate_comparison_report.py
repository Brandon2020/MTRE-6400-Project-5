"""
Generate Comparison Report - TD3 vs SAC+HER
Creates performance table and saves figures for the IEEE report.
"""

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3, SAC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ==================== 1. Setup Environment ====================
# Register AntMaze environments and create the evaluation environment
gym.register_envs(gymnasium_robotics)
env = gym.make("AntMaze_Medium-v5")

print("🔄 Loading models and running full evaluation...")

# ==================== 2. Load Trained Models ====================
# Load the best available models from previous training runs
td3_model = TD3.load("models/td3_saved_on_interrupt.zip", env=env)
print("✅ TD3 model loaded")

sac_path = "models/best_model.zip" if os.path.exists("models/best_model.zip") else "models/sac_her_final.zip"
sac_model = SAC.load(sac_path, env=env)
print("✅ SAC+HER model loaded")

# ==================== 3. Evaluation Function ====================
# This function runs multiple episodes and collects key metrics
# We need Success Rate, Average Return, Episode Length, and Goal Distance
def evaluate_model(model, n_episodes=30, name=""):
    print(f"\nEvaluating {name} on {n_episodes} episodes...")
    returns = []
    lengths = []
    successes = []
    distances = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_return = 0
        ep_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated

        success = info.get("success", False) or ep_return > 0
        final_dist = np.linalg.norm(obs['observation'][:2] - obs['desired_goal'][:2])

        returns.append(ep_return)
        lengths.append(ep_length)
        successes.append(success)
        distances.append(final_dist)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:2d}/{n_episodes} | Success: {success} | Return: {ep_return:.1f}")

    return {
        "Algorithm": name,
        "Success Rate (%)": round(np.mean(successes) * 100, 1),
        "Avg Return": round(np.mean(returns), 2),
        "Avg Episode Length": round(np.mean(lengths), 1),
        "Avg Final Goal Distance": round(np.mean(distances), 3),
    }

# Run evaluations
td3_res = evaluate_model(td3_model, 30, "TD3")
sac_res = evaluate_model(sac_model, 30, "SAC+HER")

# ==================== 4. Create Comparison Table ====================
df = pd.DataFrame([td3_res, sac_res])
print("\n" + "="*70)
print("📊 TD3 vs SAC+HER COMPARISON TABLE")
print("="*70)
print(df.to_string(index=False))

# Save table as image for IEEE report
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.3, 2)
plt.title("Performance Comparison: TD3 vs SAC+HER on AntMaze_Medium-v5", fontsize=14, pad=20)
plt.savefig("comparison_table.png", dpi=300, bbox_inches='tight')
print("✅ Table saved as comparison_table.png")

print("\n🎉 All files generated for your report!")
