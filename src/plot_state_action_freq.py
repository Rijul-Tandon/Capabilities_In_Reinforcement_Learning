import argparse
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from dqn_common import QNetwork, make_env, action_names

def plot_frequencies(env_id, model_path, episodes=100, seed=1, hidden_size=256, action_set="task"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup environment
    env = make_env(env_id, seed, action_set)
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    num_actions = env.action_space.n
    names = action_names(env_id, action_set, num_actions)

    # Load model
    q_net = QNetwork(obs_dim, num_actions, hidden_size).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    # Get grid size from unwrapped env
    width = env.unwrapped.width
    height = env.unwrapped.height

    # Initialize tracking structures
    # state_action_counts[x][y] = array of shape (num_actions,)
    state_action_counts = np.zeros((width, height, num_actions), dtype=int)
    visit_counts = np.zeros((width, height), dtype=int)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            # Extract agent position from unwrapped env
            agent_pos = env.unwrapped.agent_pos
            if agent_pos is not None:
                x, y = agent_pos
                visit_counts[x, y] += 1

                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())

                state_action_counts[x, y, action] += 1

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Visit Frequency Heatmap
    im1 = axes[0].imshow(visit_counts.T, origin="upper", cmap="YlOrRd")
    axes[0].set_title(f"{env_id} - State Visit Frequencies\n({episodes} episodes)")
    fig.colorbar(im1, ax=axes[0])
    
    axes[1].imshow(visit_counts.T, origin="upper", cmap="Blues", alpha=0.3)
    axes[1].set_title("Most Frequent Action (Count)")

    for x in range(width):
        for y in range(height):
            if visit_counts[x, y] > 0:
                best_action_idx = int(np.argmax(state_action_counts[x, y]))
                best_action_name = names[best_action_idx]
                action_count = state_action_counts[x, y, best_action_idx]
                display_text = f"{best_action_name}\n{action_count}"
                axes[1].text(x, y, display_text, ha="center", va="center", fontsize=9, fontweight="bold")

    # Add gridlines
    for ax in axes:
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(np.arange(0, width, 1))
        ax.set_yticks(np.arange(0, height, 1))

    plt.tight_layout()
    output_path = Path("plots") / f"{env_id}_state_action_freq.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--model-path", type=str, required=True, help="Path to q_net.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    args = parser.parse_args()

    plot_frequencies(
        env_id=args.env_id,
        model_path=args.model_path,
        episodes=args.episodes,
        action_set=args.action_set
    )
