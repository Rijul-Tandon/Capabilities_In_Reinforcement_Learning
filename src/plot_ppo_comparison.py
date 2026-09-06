import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def rolling_mean(data, window=10):
    if len(data) == 0:
        return data
    out = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out.append(np.mean(data[start : i + 1]))
    return np.array(out)

def load_ppo_runs(results_dir, env_id):
    runs = {"ppo_baseline": {}, "ppo_vqvae_masked": {}}
    
    results_path = Path(results_dir)
    for run_dir in results_path.glob(f"{env_id}__*"):
        returns_file = run_dir / "returns.npy"
        if not returns_file.exists():
            continue
            
        parts = run_dir.name.split("__")
        if len(parts) >= 3:
            agent = parts[1]
            try:
                seed = int(parts[2])
            except ValueError:
                seed = 1
                
            if agent in runs:
                returns = np.load(returns_file)
                runs[agent][seed] = returns
                
    return runs

def plot_ppo_comparison(runs, env_id, output_dir, window=10):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    colors = {
        "ppo_baseline": "#1f77b4",       # Blue
        "ppo_vqvae_masked": "#2ca02c",   # Green
    }
    
    labels = {
        "ppo_baseline": "PPO Baseline",
        "ppo_vqvae_masked": "PPO + VQ-VAE Action Masking",
    }
    
    for agent, seed_dict in runs.items():
        if not seed_dict:
            continue
            
        all_series = []
        max_len = max(len(arr) for arr in seed_dict.values())
        
        for seed, arr in seed_dict.items():
            smoothed = rolling_mean(arr, window=window)
            all_series.append(smoothed)
            
        # Pad series to equal length for averaging
        padded_series = []
        for s in all_series:
            if len(s) < max_len:
                padded = np.pad(s, (0, max_len - len(s)), mode='edge')
            else:
                padded = s
            padded_series.append(padded)
            
        padded_series = np.array(padded_series)
        mean_curve = np.mean(padded_series, axis=0)
        std_curve = np.std(padded_series, axis=0) if len(padded_series) > 1 else np.zeros_like(mean_curve)
        
        episodes = np.arange(1, max_len + 1)
        ax.plot(episodes, mean_curve, label=f"{labels[agent]} ({len(seed_dict)} seeds)", color=colors[agent], linewidth=2.0)
        if len(seed_dict) > 1:
            ax.fill_between(episodes, mean_curve - std_curve * 0.5, mean_curve + std_curve * 0.5, color=colors[agent], alpha=0.15)
            
    ax.set_title(f"MuJoCo Performance ({env_id}) — PPO Baseline vs Action Masked", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Completed Episodes", fontsize=9.5, fontweight="medium")
    ax.set_ylabel("Episodic Return", fontsize=9.5, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="best", frameon=True)
    
    output_path = Path(output_dir) / f"{env_id}_ppo_comparison.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PPO comparison plot → {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots/ppo_comparison")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()
    
    runs = load_ppo_runs(args.results_dir, args.env_id)
    plot_ppo_comparison(runs, args.env_id, args.output_dir, window=args.window)

if __name__ == "__main__":
    main()
