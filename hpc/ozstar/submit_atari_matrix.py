"""Submit one OzSTAR Slurm job for every requested environment, agent, and seed."""

import argparse
import subprocess
from pathlib import Path


ENVIRONMENTS = [
    "ALE/DemonAttack-v5",
    "ALE/BeamRider-v5",
    "ALE/MsPacman-v5",
    "ALE/Qbert-v5",
]
AGENTS = [
    "ddqn_baseline",
    "ddqn_vqvae_masked",
    "dqn_baseline_standard",
]


def short_name(value: str) -> str:
    return (
        value
        .replace("ddqn_", "d")
        .replace("dqn_", "q")
        .replace("vqvae_masked", "mask")
        .replace("baseline_standard", "std")
        .replace("baseline", "base")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=ENVIRONMENTS)
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=AGENTS)
    parser.add_argument("--time", default="1-00:00:00", help="Slurm wall-time limit, e.g. 2-00:00:00")
    parser.add_argument("--gres", default="gpu:1", help="Use gpu:a100:1 only if that GRES name is available.")
    parser.add_argument("--partition", default=None, help="Optional; OzSTAR recommends selecting the cluster via submit host instead.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    script_dir = Path(__file__).resolve().parent
    batch_script = script_dir / "train_atari.slurm"
    log_dir = script_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if not batch_script.exists():
        raise FileNotFoundError(batch_script)

    for env_id in args.envs:
        env_name = env_id.split("/", 1)[1].replace("-v5", "").lower()
        for agent in args.agents:
            for seed in args.seeds:
                job_name = f"a-{env_name[:5]}-{short_name(agent)[:5]}-{seed}"  # Slurm-safe, <=15 chars.
                command = [
                    "sbatch", f"--job-name={job_name}", f"--time={args.time}", f"--gres={args.gres}",
                    f"--output={log_dir}/%x-%j.out", f"--export=ALL,PROJECT_DIR={project_dir}",
                ]
                if args.partition:
                    command.append(f"--partition={args.partition}")
                command.extend([str(batch_script), agent, env_id, str(seed), str(args.timesteps)])
                print(" ".join(command))
                if not args.dry_run:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
