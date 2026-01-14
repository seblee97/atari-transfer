#!/usr/bin/env python3
import argparse
import os
import json
from itertools import combinations

def generate_slurm_script(
    job_name,
    algorithm,
    source_game,
    target_game,
    source_timesteps,
    target_timesteps,
    output_dir,
    partition="gpu",
    time_limit="24:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    conda_env=None,
    venv_path=None,
    freeze_encoder=False,
    reinit_head=False,
):
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_logs/{job_name}_%j.out
#SBATCH --error={output_dir}/slurm_logs/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}

echo "Starting job: {job_name}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Algorithm: {algorithm}"
echo "Source game: {source_game}"
echo "Target game: {target_game}"

"""

    if venv_path:
        script_content += f"""
# Activate virtual environment
source {venv_path}/bin/activate
"""
    elif conda_env:
        script_content += f"""
# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {conda_env}
"""

    script_content += f"""
# Run transfer learning
python transfer_learning.py \\
    --algorithm {algorithm} \\
    --source-game {source_game} \\
    --target-game {target_game} \\
    --source-timesteps {source_timesteps} \\
    --target-timesteps {target_timesteps} \\
    --output-dir {output_dir}"""

    if freeze_encoder:
        script_content += " \\\n    --freeze-encoder"
    if reinit_head:
        script_content += " \\\n    --reinit-head"

    script_content += """

echo "Job completed: $(date)"
"""

    return script_content

def generate_all_jobs(
    games,
    algorithms,
    source_timesteps,
    target_timesteps,
    output_dir,
    partition="gpu",
    time_limit="24:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    conda_env=None,
    venv_path=None,
    freeze_encoder=False,
    reinit_head=False,
):
    os.makedirs(output_dir, exist_ok=True)
    scripts_dir = os.path.join(output_dir, "slurm_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "slurm_logs"), exist_ok=True)

    game_pairs = list(combinations(games, 2))

    all_jobs = []

    for algorithm in algorithms:
        for source_game, target_game in game_pairs:
            for direction in [True, False]:
                if direction:
                    src, tgt = source_game, target_game
                else:
                    src, tgt = target_game, source_game

                job_name = f"{algorithm}_{src}_to_{tgt}"
                script_path = os.path.join(scripts_dir, f"{job_name}.sh")

                script_content = generate_slurm_script(
                    job_name=job_name,
                    algorithm=algorithm,
                    source_game=src,
                    target_game=tgt,
                    source_timesteps=source_timesteps,
                    target_timesteps=target_timesteps,
                    output_dir=output_dir,
                    partition=partition,
                    time_limit=time_limit,
                    mem=mem,
                    cpus=cpus,
                    gpus=gpus,
                    conda_env=conda_env,
                    venv_path=venv_path,
                    freeze_encoder=freeze_encoder,
                    reinit_head=reinit_head,
                )

                with open(script_path, "w") as f:
                    f.write(script_content)

                os.chmod(script_path, 0o755)

                all_jobs.append({
                    "job_name": job_name,
                    "script_path": script_path,
                    "algorithm": algorithm,
                    "source_game": src,
                    "target_game": tgt,
                })

                print(f"Generated: {script_path}")

    submit_script_path = os.path.join(scripts_dir, "submit_all.sh")
    with open(submit_script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Submit all SLURM jobs\n\n")
        for job in all_jobs:
            f.write(f"sbatch {job['script_path']}\n")

    os.chmod(submit_script_path, 0o755)

    jobs_summary_path = os.path.join(output_dir, "jobs_summary.json")
    with open(jobs_summary_path, "w") as f:
        json.dump({
            "total_jobs": len(all_jobs),
            "games": games,
            "algorithms": algorithms,
            "game_pairs": [[src, tgt] for src, tgt in game_pairs],
            "jobs": all_jobs,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_jobs)} SLURM job scripts")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Submit all jobs: {submit_script_path}")
    print(f"Jobs summary: {jobs_summary_path}")
    print(f"{'='*60}\n")
    print(f"To submit all jobs, run:")
    print(f"  bash {submit_script_path}")
    print(f"\nOr submit individual jobs:")
    print(f"  sbatch {scripts_dir}/<job_name>.sh")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM job scripts for transfer learning experiments")
    parser.add_argument("--config", type=str, help="JSON config file with games list")
    parser.add_argument("--games", type=str, nargs="+", help="List of Atari games (e.g., Pong Breakout SpaceInvaders)")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None,
                        choices=["dqn", "ppo", "sac", "qrdqn"], help="RL algorithms to use")
    parser.add_argument("--source-timesteps", type=int, default=None,
                        help="Training timesteps for source game")
    parser.add_argument("--target-timesteps", type=int, default=None,
                        help="Training timesteps for target game")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for experiments")
    parser.add_argument("--partition", type=str, default=None,
                        help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default=None,
                        help="Time limit per job (HH:MM:SS)")
    parser.add_argument("--mem", type=str, default=None,
                        help="Memory per job")
    parser.add_argument("--cpus", type=int, default=None,
                        help="CPUs per task")
    parser.add_argument("--gpus", type=int, default=None,
                        help="GPUs per task")
    parser.add_argument("--conda-env", type=str, default=None,
                        help="Conda environment name to activate")
    parser.add_argument("--venv-path", type=str, default=None,
                        help="Path to virtual environment (e.g., /path/to/venv)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze CNN encoder during target game training")
    parser.add_argument("--reinit-head", action="store_true",
                        help="Reinitialize head layers before target game training")

    args = parser.parse_args()

    # Set defaults for config-based parameters
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        games = config.get("games", [])

        # Load config values, allowing command-line to override
        if "training" in config:
            training_config = config["training"]
            if args.source_timesteps is None:
                args.source_timesteps = training_config.get("source_timesteps", 1000000)
            if args.target_timesteps is None:
                args.target_timesteps = training_config.get("target_timesteps", 1000000)
            # For boolean flags, only override if not set on command line
            if not args.freeze_encoder:
                args.freeze_encoder = training_config.get("freeze_encoder", False)
            if not args.reinit_head:
                args.reinit_head = training_config.get("reinit_head", False)

        if "slurm" in config:
            slurm_config = config["slurm"]
            if args.partition is None:
                args.partition = slurm_config.get("partition", "gpu")
            if args.time_limit is None:
                args.time_limit = slurm_config.get("time_limit", "24:00:00")
            if args.mem is None:
                args.mem = slurm_config.get("mem", "32G")
            if args.cpus is None:
                args.cpus = slurm_config.get("cpus", 4)
            if args.gpus is None:
                args.gpus = slurm_config.get("gpus", 1)
            if args.conda_env is None:
                args.conda_env = slurm_config.get("conda_env")
            if args.venv_path is None:
                args.venv_path = slurm_config.get("venv_path")

        if "algorithms" in config and args.algorithms is None:
            args.algorithms = config["algorithms"]
    elif args.games:
        games = args.games
    else:
        print("Error: Must provide either --config or --games")
        exit(1)

    # Apply final defaults for any parameters still None
    if args.algorithms is None:
        args.algorithms = ["dqn", "ppo"]
    if args.source_timesteps is None:
        args.source_timesteps = 1000000
    if args.target_timesteps is None:
        args.target_timesteps = 1000000
    if args.partition is None:
        args.partition = "gpu"
    if args.time_limit is None:
        args.time_limit = "24:00:00"
    if args.mem is None:
        args.mem = "32G"
    if args.cpus is None:
        args.cpus = 4
    if args.gpus is None:
        args.gpus = 1

    if len(games) < 2:
        print("Error: Need at least 2 games for pairwise transfer learning")
        exit(1)

    generate_all_jobs(
        games=games,
        algorithms=args.algorithms,
        source_timesteps=args.source_timesteps,
        target_timesteps=args.target_timesteps,
        output_dir=args.output_dir,
        partition=args.partition,
        time_limit=args.time_limit,
        mem=args.mem,
        cpus=args.cpus,
        gpus=args.gpus,
        conda_env=args.conda_env,
        venv_path=args.venv_path,
        freeze_encoder=args.freeze_encoder,
        reinit_head=args.reinit_head,
    )
