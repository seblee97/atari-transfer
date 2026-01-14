#!/usr/bin/env python3
"""
Generate SLURM job scripts for transfer learning experiments using pre-trained models.

This script generates SLURM jobs that:
1. Download pre-trained models from RL Baselines3 Zoo
2. Run transfer learning using pre-trained source models
3. Skip source training entirely, saving ~50% time per experiment
"""

import os
import json
import argparse
from itertools import combinations


def create_slurm_script_pretrained(
    job_name,
    algorithm,
    source_game,
    target_game,
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
    """Create a SLURM script for transfer learning with pre-trained source model."""

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_logs/{job_name}_%j.out
#SBATCH --error={output_dir}/slurm_logs/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to project directory
cd $SLURM_SUBMIT_DIR
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
# Install requests if not already installed (needed for download script)
pip install -q requests 2>/dev/null || true

# Download pre-trained model if not already downloaded
PRETRAINED_DIR="pretrained_models/{algorithm}"
PRETRAINED_MODEL="$PRETRAINED_DIR/{source_game}.zip"

echo "Checking for pre-trained model: $PRETRAINED_MODEL"

if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Pre-trained model not found. Downloading from RL Baselines3 Zoo..."
    python download_pretrained_models.py \\
        --algorithm {algorithm} \\
        --game {source_game} \\
        --output-dir pretrained_models

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download pre-trained model"
        echo "Game '{source_game}' may not be available in the zoo"
        echo "Falling back to training from scratch..."

        # Fall back to regular transfer learning
        python transfer_learning.py \\
            --algorithm {algorithm} \\
            --source-game {source_game} \\
            --target-game {target_game} \\
            --source-timesteps {target_timesteps} \\
            --target-timesteps {target_timesteps} \\
            --output-dir {output_dir}"""

    if freeze_encoder:
        script_content += " \\\n            --freeze-encoder"
    if reinit_head:
        script_content += " \\\n            --reinit-head"

    script_content += """

        exit $?
    fi
fi

echo "Using pre-trained model: $PRETRAINED_MODEL"

# Run transfer learning with pre-trained model
python transfer_learning_pretrained.py \\
    --algorithm {algorithm} \\
    --source-game {source_game} \\
    --target-game {target_game} \\
    --pretrained-model "$PRETRAINED_MODEL" \\
    --target-timesteps {target_timesteps} \\
    --output-dir {output_dir}""".format(
        algorithm=algorithm,
        source_game=source_game,
        target_game=target_game,
        target_timesteps=target_timesteps,
        output_dir=output_dir
    )

    if freeze_encoder:
        script_content += " \\\n    --freeze-encoder"
    if reinit_head:
        script_content += " \\\n    --reinit-head"

    script_content += """

echo "Job completed: $(date)"
"""

    return script_content


def generate_all_jobs_pretrained(
    games,
    algorithms,
    target_timesteps,
    output_dir,
    partition="gpu",
    time_limit="12:00:00",  # Shorter since no source training
    mem="32G",
    cpus=4,
    gpus=1,
    conda_env=None,
    venv_path=None,
    freeze_encoder=False,
    reinit_head=False,
):
    """Generate SLURM jobs for all pairwise transfer learning experiments with pre-trained models."""

    os.makedirs(output_dir, exist_ok=True)
    scripts_dir = os.path.join(output_dir, "slurm_scripts_pretrained")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "slurm_logs"), exist_ok=True)

    game_pairs = list(combinations(games, 2))

    all_jobs = []

    # Generate jobs for each algorithm and game pair (both directions)
    for algorithm in algorithms:
        for game1, game2 in game_pairs:
            # game1 -> game2
            for source, target in [(game1, game2), (game2, game1)]:
                job_name = f"{algorithm}_{source}_to_{target}_pretrained"
                script_path = os.path.join(scripts_dir, f"{job_name}.sh")

                script_content = create_slurm_script_pretrained(
                    job_name=job_name,
                    algorithm=algorithm,
                    source_game=source,
                    target_game=target,
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
                    "source": source,
                    "target": target,
                })

    # Create a submit_all script
    submit_script_path = os.path.join(scripts_dir, "submit_all.sh")
    with open(submit_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all transfer learning jobs with pre-trained models\n\n")
        for job in all_jobs:
            f.write(f"sbatch {job['script_path']}\n")
        f.write("\necho 'Submitted {0} jobs'\n".format(len(all_jobs)))

    os.chmod(submit_script_path, 0o755)

    # Create a download_all_models script
    download_script_path = os.path.join(scripts_dir, "download_all_models.sh")
    with open(download_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Pre-download all models before submitting jobs\n\n")

        if venv_path:
            f.write(f"source {venv_path}/bin/activate\n\n")
        elif conda_env:
            f.write(f"source $(conda info --base)/etc/profile.d/conda.sh\n")
            f.write(f"conda activate {conda_env}\n\n")

        f.write("pip install -q requests 2>/dev/null || true\n\n")
        f.write(f"python download_pretrained_models.py \\\n")
        f.write(f"    --games {' '.join(games)} \\\n")
        f.write(f"    --algorithms {' '.join(algorithms)} \\\n")
        f.write(f"    --output-dir pretrained_models\n")

    os.chmod(download_script_path, 0o755)

    # Print summary
    print("\n" + "=" * 80)
    print("SLURM JOB GENERATION WITH PRE-TRAINED MODELS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(all_jobs)} jobs")
    print(f"Scripts directory: {scripts_dir}")
    print(f"\nGames: {', '.join(games)}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Target timesteps: {target_timesteps}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Reinitialize head: {reinit_head}")

    print("\n" + "=" * 80)
    print("RECOMMENDED WORKFLOW")
    print("=" * 80)
    print("\n1. PRE-DOWNLOAD ALL MODELS (recommended, saves job startup time):")
    print(f"   bash {download_script_path}")

    print("\n2. SUBMIT ALL JOBS:")
    print(f"   bash {submit_script_path}")

    print("\n   OR submit individual jobs:")
    print(f"   sbatch {scripts_dir}/<job_name>.sh")

    print("\n3. MONITOR JOBS:")
    print("   squeue -u $USER")

    print("\n4. CHECK LOGS:")
    print(f"   tail -f {output_dir}/slurm_logs/<job_name>_<job_id>.out")

    print("\n" + "=" * 80)
    print("BENEFITS OF USING PRE-TRAINED MODELS")
    print("=" * 80)
    print(f"\nTime savings per experiment:")
    print(f"  - Without pre-trained: ~4-8 hours (source + target training)")
    print(f"  - With pre-trained: ~2-4 hours (target training only)")
    print(f"  - Savings: ~50% time reduction per experiment")
    print(f"\nTotal experiments: {len(all_jobs)}")
    print(f"Estimated time saved: ~{len(all_jobs) * 3} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SLURM job scripts for transfer learning with pre-trained models"
    )
    parser.add_argument("--config", type=str, help="JSON config file with games list")
    parser.add_argument("--games", type=str, nargs="+",
                       help="List of Atari games (e.g., Pong Breakout SpaceInvaders)")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None,
                       choices=["dqn", "ppo", "qrdqn"],
                       help="RL algorithms to use (only dqn, ppo, qrdqn have pre-trained models)")
    parser.add_argument("--target-timesteps", type=int, default=None,
                       help="Training timesteps for target game")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for experiments")
    parser.add_argument("--partition", type=str, default=None,
                       help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default=None,
                       help="Time limit per job (HH:MM:SS) - default 12:00:00 (shorter since no source training)")
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

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        games = config.get("games", [])

        # Load config values, allowing command-line to override
        if "training" in config:
            training_config = config["training"]
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
                args.time_limit = slurm_config.get("time_limit", "12:00:00")
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
            # Filter to only algorithms with pre-trained models available
            available_algos = ["dqn", "ppo", "qrdqn"]
            args.algorithms = [a for a in config["algorithms"] if a in available_algos]
    elif args.games:
        games = args.games
    else:
        print("Error: Must provide either --config or --games")
        exit(1)

    # Apply final defaults for any parameters still None
    if args.algorithms is None:
        args.algorithms = ["dqn", "ppo"]  # Default to algorithms with most pre-trained models
    if args.target_timesteps is None:
        args.target_timesteps = 1000000
    if args.partition is None:
        args.partition = "gpu"
    if args.time_limit is None:
        args.time_limit = "12:00:00"  # Shorter default since no source training
    if args.mem is None:
        args.mem = "32G"
    if args.cpus is None:
        args.cpus = 4
    if args.gpus is None:
        args.gpus = 1

    # Validate algorithms
    valid_algos = ["dqn", "ppo", "qrdqn"]
    for algo in args.algorithms:
        if algo not in valid_algos:
            print(f"Warning: Algorithm '{algo}' does not have pre-trained models available in RL Baselines3 Zoo")
            print(f"Available algorithms with pre-trained models: {', '.join(valid_algos)}")
            print("This algorithm will fall back to training from scratch.")

    if len(games) < 2:
        print("Error: Need at least 2 games for pairwise transfer learning")
        exit(1)

    generate_all_jobs_pretrained(
        games=games,
        algorithms=args.algorithms,
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
