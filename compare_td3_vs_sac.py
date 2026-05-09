"""
Side-by-side Trajectory Comparison Plot
"""
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3, SAC
import matplotlib.pyplot as plt
import numpy as np
import os

gym.register_envs(gymnasium_robotics)
env = gym.make("AntMaze_Medium-v5")

td3_model = TD3.load("models/td3_saved_on_interrupt.zip", env=env)
sac_model = SAC.load("models/best_model.zip" if os.path.exists("models/best_model.zip") else "models/sac_her_final.zip", env=env)

def get_trajectory(model, seed=42):
    obs, _ = env.reset(seed=seed)
    positions = [obs['observation'][:2]]
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        positions.append(obs['observation'][:2])
        if info.get("success", False) or terminated or truncated:
            break
    return np.array(positions)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
for ax, model, title, color in zip([ax1, ax2], [td3_model, sac_model], ["TD3", "SAC+HER"], ['red', 'green']):
    pos = get_trajectory(model)
    walls = [[[0,0],[0,6]], [[0,6],[6,6]], [[6,6],[6,0]], [[6,0],[0,0]],
             [[1,0],[1,4]], [[2,3],[2,6]], [[3,1],[3,3]], [[4,0],[4,2]],
             [[4,4],[4,6]], [[5,2],[5,4]]]
    for w in walls:
        ax.plot([w[0][0], w[1][0]], [w[0][1], w[1][1]], 'orange', linewidth=8)
    ax.plot(5.5, 5.5, marker='*', markersize=25, color='white', markeredgecolor='black', markeredgewidth=2)
    ax.plot(pos[:,0], pos[:,1], color=color, linewidth=2.5, label=title + " Path")
    ax.scatter(pos[0,0], pos[0,1], color='blue', s=100)
    ax.set_title(title + " Trajectory")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("TD3 vs SAC+HER Trajectory Comparison")
plt.tight_layout()
plt.savefig("td3_vs_sac_her_trajectories.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ Comparison plot saved!")
