Make a directory antmaze_project
cd ~/antmaze_project
This is where you will put all your code
source rl_env/bin/activate
before running python train_sac_her.py



cd ~/antmaze_project
cat > README.md << 'EOF'
# AntMaze Deep Reinforcement Learning Project
**MTRE 6400/8400 Project #5** - Deep Reinforcement Learning for Mobile Robot Navigation

This project implements and compares **TD3** and **SAC+HER** on the `AntMaze_Medium-v5` environment using Stable-Baselines3.

---

## Project Structure
antmaze_project/

├── train_td3_basic.py              # TD3 Training Script

├── train_sac_her.py                # SAC + HER Training Script

├── generate_comparison_report.py   # Evaluation + Comparison Table

├── compare_td3_vs_sac.py           # Side-by-side Trajectory Plot

├── plot_td3.py                     # TD3 Trajectory Plot

├── README.md

├── models/                         # Trained models (best_model.zip, etc.)

├── logs/                           # TensorBoard logs

└── *.png                           # Generated figures for report




---

## Setup (Already Done)

cd ~/antmaze_project
source rl_env/bin/activate

1. Training TD3 (Basic)
Train
Bash python train_td3_basic.py

Press Ctrl + C to stop gracefully.
Best model saved automatically as models/best_model.zip

Generate TD3 Trajectory Plot
Bashpython plot_td3.py

2. Training SAC + HER (Recommended / Stronger)
Train
Bash python train_sac_her.py

Uses Hindsight Experience Replay (HER) — much better for sparse rewards.
Recommended to run for 500k–1M+ timesteps.


3. Generate Comparison Report (Best for Report)
Bash python generate_comparison_report.py
This creates:

comparison_table.png → Performance comparison table
td3_vs_sac_her_trajectories.png → Side-by-side trajectory map


4. Live Demo (Watch the Ant Move)
Bash python -c '
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
import os
gym.register_envs(gymnasium_robotics)
env = gym.make("AntMaze_Medium-v5", render_mode="human")
model = SAC.load("models/best_model.zip" if os.path.exists("models/best_model.zip") else "models/sac_her_final.zip", env=env)
print("Live Demo Started - Watch the window!")
obs, _ = env.reset(seed=42)
for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, info = env.step(action)
    if info.get("success", False): 
        print("Goal Reached!")
        break
env.close()
'

5. TensorBoard (Live Learning Curves)
Bash tensorboard --logdir ./logs --host 0.0.0.0
→ Open in browser: http://localhost:6006

6. Copy Files to Windows
Bash cp *.png /mnt/c/Users/bsnap/Videos/ 2>/dev/null || cp *.png /mnt/c/Users/bsnap/Desktop/
explorer.exe /mnt/c/Users/bsnap/Videos

AI Usage Statement (Add to your report)
AI tools (Grok by xAI) were used for:
Writing and debugging training scripts (TD3 and SAC+HER)
Generating commented code and README
Creating visualization and comparison plots
Drafting report sections and explanations
All code was reviewed, tested, and understood by the author.
