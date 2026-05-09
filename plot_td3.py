"""
TD3 Individual Trajectory Plot
Creates a clean single plot for TD3 path visualization.
"""

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
import numpy as np
import os

gym.register_envs(gymnasium_robotics)
env = gym.make("AntMaze_Medium-v5")

# Load TD3 model
model = TD3.load("models/td3_saved_on_interrupt.zip", env=env)

print("🎨 Generating TD3 Trajectory Plot...")

# Record one episode trajectory
obs, _ = env.reset(seed=42)
positions = [obs['observation'][:2]]

for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, info = env.step(action)
    positions.append(obs['observation'][:2])
    if info.get("success", False) or terminated or truncated:
        break

positions = np.array(positions)

# ==================== Plotting ====================
plt.figure(figsize=(10, 10))
plt.title("TD3 Trajectory on AntMaze_Medium-v5", fontsize=14)

# Draw maze walls in orange
walls = [[[0,0],[0,6]], [[0,6],[6,6]], [[6,6],[6,0]], [[6,0],[0,0]],
         [[1,0],[1,4]], [[2,3],[2,6]], [[3,1],[3,3]], [[4,0],[4,2]],
         [[4,4],[4,6]], [[5,2],[5,4]]]
for w in walls:
    plt.plot([w[0][0], w[1][0]], [w[0][1], w[1][1]], 'orange', linewidth=8)

# Goal (white star)
plt.plot(5.5, 5.5, marker='*', markersize=25, color='white', 
         markeredgecolor='black', markeredgewidth=2, label='Goal')

# TD3 path
plt.plot(positions[:,0], positions[:,1], 'red', linewidth=2.8, label='TD3 Path')
plt.scatter(positions[0,0], positions[0,1], color='blue', s=120, label='Start')

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("td3_trajectory.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ TD3 trajectory plot saved as td3_trajectory.png")
