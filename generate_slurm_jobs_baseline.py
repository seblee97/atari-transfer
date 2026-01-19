#!/usr/bin/env python3
"""
Generate SLURM job scripts for baseline training (from scratch, no transfer).

This script generates jobs that train models from scratch on each game to establish
baseline learning curves for comparison with transfer learning experiments.
"""

import os
import json
import argparse


def create_baseline_slurm_script(
    job_name,
    algorithm,
    game,
    timesteps,
    output_dir,
    checkpoint_freq=50000,
    eval_freq=10000,
    partition="gpu",
    time_limit="7-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    conda_env=None,
    venv_path=None,
):
    """Create a SLURM script for baseline training."""

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
echo "Activating virtual environment: {venv_path}"
if [ -f "{venv_path}/bin/activate" ]; then
    source {venv_path}/bin/activate
    echo "Virtual environment activated successfully"
    echo "Python path: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "ERROR: Virtual environment not found at {venv_path}/bin/activate"
    exit 1
fi
"""
    elif conda_env:
        script_content += f"""
# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {conda_env}
"""

    # Create experiment directory structure
    exp_name = f"{algorithm}_{game}_baseline"
    exp_dir = f"{output_dir}/{exp_name}_${{SLURM_JOB_ID}}"

    script_content += f"""
# Create experiment directories
EXP_DIR="{exp_dir}"
mkdir -p "${{EXP_DIR}}/checkpoints"
mkdir -p "${{EXP_DIR}}/logs"

echo "Experiment directory: $EXP_DIR"

# Run baseline training
python train_{algorithm}.py \\
    --game {game} \\
    --timesteps {timesteps} \\
    --checkpoint-dir "${{EXP_DIR}}/checkpoints" \\
    --log-dir "${{EXP_DIR}}/logs" \\
    --checkpoint-freq {checkpoint_freq} \\
    --eval-freq {eval_freq}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $EXIT_CODE"
fi

echo "Job completed: $(date)"
exit $EXIT_CODE
"""

    return script_content


def generate_all_baseline_jobs(
    games,
    algorithms,
    timesteps,
    output_dir,
    checkpoint_freq=50000,
    eval_freq=10000,
    partition="gpu",
    time_limit="7-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    conda_env=None,
    venv_path=None,
):
    """Generate SLURM jobs for all baseline training experiments."""

    os.makedirs(output_dir, exist_ok=True)
    scripts_dir = os.path.join(output_dir, "slurm_scripts_baseline")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "slurm_logs"), exist_ok=True)

    all_jobs = []

    # Generate jobs for each algorithm and game
    for algorithm in algorithms:
        for game in games:
            job_name = f"{algorithm}_{game}_baseline"
            script_path = os.path.join(scripts_dir, f"{job_name}.sh")

            script_content = create_baseline_slurm_script(
                job_name=job_name,
                algorithm=algorithm,
                game=game,
                timesteps=timesteps,
                output_dir=output_dir,
                checkpoint_freq=checkpoint_freq,
                eval_freq=eval_freq,
                partition=partition,
                time_limit=time_limit,
                mem=mem,
                cpus=cpus,
                gpus=gpus,
                conda_env=conda_env,
                venv_path=venv_path,
            )

            with open(script_path, "w") as f:
                f.write(script_content)

            os.chmod(script_path, 0o755)

            all_jobs.append({
                "job_name": job_name,
                "script_path": script_path,
                "algorithm": algorithm,
                "game": game,
            })

    # Create a submit_all script
    submit_script_path = os.path.join(scripts_dir, "submit_all.sh")
    with open(submit_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all baseline training jobs\n\n")
        for job in all_jobs:
            f.write(f"sbatch {job['script_path']}\n")
        f.write(f"\necho 'Submitted {len(all_jobs)} jobs'\n")

    os.chmod(submit_script_path, 0o755)

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE SLURM JOB GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(all_jobs)} jobs")
    print(f"Scripts directory: {scripts_dir}")
    print(f"\nGames: {', '.join(games)}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Training timesteps: {timesteps}")

    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)
    print("\n1. SUBMIT ALL JOBS:")
    print(f"   bash {submit_script_path}")

    print("\n   OR submit individual jobs:")
    print(f"   sbatch {scripts_dir}/<job_name>.sh")

    print("\n2. MONITOR JOBS:")
    print("   squeue -u $USER")

    print("\n3. CHECK LOGS:")
    print(f"   tail -f {output_dir}/slurm_logs/<job_name>_<job_id>.out")

    print("\n" + "=" * 80)
    print("PURPOSE")
    print("=" * 80)
    print("\nThese baseline jobs train models from scratch to establish learning curves")
    print("for comparison with transfer learning experiments.")
    print(f"\nResults will be saved to: {output_dir}/")
    print("Each experiment will have:")
    print("  - checkpoints/  (model checkpoints)")
    print("  - logs/         (TensorBoard logs with learning curves)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SLURM job scripts for baseline training"
    )
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--games", type=str, nargs="+",
                       help="List of games")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       choices=["dqn", "ppo", "qrdqn", "sac"],
                       help="RL algorithms to use")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Training timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                       help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=None,
                       help="Evaluation frequency")
    parser.add_argument("--output-dir", type=str, default="results_baseline",
                       help="Output directory for experiments")
    parser.add_argument("--partition", type=str, default=None,
                       help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default=None,
                       help="Time limit per job (format: D-HH:MM:SS)")
    parser.add_argument("--mem", type=str, default=None,
                       help="Memory per job")
    parser.add_argument("--cpus", type=int, default=None,
                       help="CPUs per task")
    parser.add_argument("--gpus", type=int, default=None,
                       help="GPUs per task")
    parser.add_argument("--conda-env", type=str, default=None,
                       help="Conda environment name to activate")
    parser.add_argument("--venv-path", type=str, default=None,
                       help="Path to virtual environment")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        games = config.get("games", [])

        # Load config values, allowing command-line to override
        if "training" in config:
            training_config = config["training"]
            if args.timesteps is None:
                args.timesteps = training_config.get("timesteps", 1000000)
            if args.checkpoint_freq is None:
                args.checkpoint_freq = training_config.get("checkpoint_freq", 50000)
            if args.eval_freq is None:
                args.eval_freq = training_config.get("eval_freq", 10000)

        if "slurm" in config:
            slurm_config = config["slurm"]
            if args.partition is None:
                args.partition = slurm_config.get("partition", "gpu")
            if args.time_limit is None:
                args.time_limit = slurm_config.get("time_limit", "7-00:00:00")
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

        # Load output_dir from config if not provided on command line
        if args.output_dir == "results_baseline" and "output_dir" in config:
            args.output_dir = config["output_dir"]

    elif args.games:
        games = args.games
    else:
        print("Error: Must provide either --config or --games")
        exit(1)

    # Apply final defaults
    if args.algorithms is None:
        args.algorithms = ["dqn", "ppo", "qrdqn"]
    if args.timesteps is None:
        args.timesteps = 1000000
    if args.checkpoint_freq is None:
        args.checkpoint_freq = 50000
    if args.eval_freq is None:
        args.eval_freq = 10000
    if args.partition is None:
        args.partition = "gpu"
    if args.time_limit is None:
        args.time_limit = "7-00:00:00"
    if args.mem is None:
        args.mem = "32G"
    if args.cpus is None:
        args.cpus = 4
    if args.gpus is None:
        args.gpus = 1

    if len(games) == 0:
        print("Error: No games specified")
        exit(1)

    generate_all_baseline_jobs(
        games=games,
        algorithms=args.algorithms,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        partition=args.partition,
        time_limit=args.time_limit,
        mem=args.mem,
        cpus=args.cpus,
        gpus=args.gpus,
        conda_env=args.conda_env,
        venv_path=args.venv_path,
    )
