#!/usr/bin/env python3
"""
Generate SLURM job scripts for phased transfer learning experiments (Version 2).

This version uses CONTINUOUS source training (one job per source game) instead of
splitting source training into multiple phases. This avoids checkpoint resume issues
and ensures smooth learning curves.

Key improvements:
- Source training is ONE continuous job that saves checkpoints at intervals
- Transfer jobs depend on source job but can start before source completes
- Transfer jobs wait for specific checkpoint files before starting training
- No model reloading between phases = no performance drops
"""
import os
import json
import argparse
from pathlib import Path


def create_continuous_source_slurm_script(
    job_name,
    algorithm,
    game,
    total_timesteps,
    checkpoint_intervals,
    output_dir,
    eval_freq=10000,
    partition="gpu",
    time_limit="7-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    venv_path=None,
):
    """Create a SLURM script for continuous source training with checkpoints."""

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

# Project directory
PROJECT_DIR=$SLURM_SUBMIT_DIR
cd $PROJECT_DIR
echo "Project directory: $PROJECT_DIR"
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

    exp_name = f"{algorithm}_{game}_source"
    checkpoint_intervals_str = " ".join(str(x) for x in checkpoint_intervals)

    script_content += f"""
# Determine absolute output directory
if [[ "{output_dir}" = /* ]]; then
    OUTPUT_BASE="{output_dir}"
else
    OUTPUT_BASE="$PROJECT_DIR/{output_dir}"
fi

# Create experiment directories
EXP_DIR="${{OUTPUT_BASE}}/{exp_name}"
mkdir -p "${{EXP_DIR}}/checkpoints"
mkdir -p "${{EXP_DIR}}/logs"

echo "Experiment directory: $EXP_DIR"
echo "Training for {total_timesteps:,} timesteps continuously"
echo "Checkpoints at: {checkpoint_intervals_str}"

# Run continuous training with checkpoints
python train_continuous_checkpoints.py \\
    --algorithm {algorithm} \\
    --game {game} \\
    --timesteps {total_timesteps} \\
    --checkpoint-dir "${{EXP_DIR}}/checkpoints" \\
    --log-dir "${{EXP_DIR}}/logs" \\
    --checkpoint-intervals {checkpoint_intervals_str} \\
    --eval-freq {eval_freq}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    echo "Final checkpoint count: $(ls ${{EXP_DIR}}/checkpoints/*.zip 2>/dev/null | wc -l)"
else
    echo "Training failed with exit code $EXIT_CODE"
fi

echo "Job completed: $(date)"
exit $EXIT_CODE
"""

    return script_content


def create_transfer_with_wait_slurm_script(
    job_name,
    algorithm,
    source_game,
    target_game,
    checkpoint_timesteps,
    total_timesteps,
    output_dir,
    source_exp_dir,
    dependency_job_id_var,
    eval_freq=10000,
    freeze_encoder=False,
    reinit_head=True,
    partition="gpu",
    time_limit="2-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    venv_path=None,
):
    """Create a SLURM script for transfer learning that waits for checkpoint."""

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_logs/{job_name}_%j.out
#SBATCH --error={output_dir}/slurm_logs/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --dependency=afterany:{dependency_job_id_var}

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Project directory
PROJECT_DIR=$SLURM_SUBMIT_DIR
cd $PROJECT_DIR
echo "Project directory: $PROJECT_DIR"
"""

    if venv_path:
        script_content += f"""
# Activate virtual environment
echo "Activating virtual environment: {venv_path}"
if [ -f "{venv_path}/bin/activate" ]; then
    source {venv_path}/bin/activate
    echo "Virtual environment activated successfully"
else
    echo "ERROR: Virtual environment not found at {venv_path}/bin/activate"
    exit 1
fi
"""

    transfer_name = f"{algorithm}_{source_game}_to_{target_game}_ckpt{checkpoint_timesteps}"

    script_content += f"""
# Determine absolute paths
if [[ "{output_dir}" = /* ]]; then
    OUTPUT_BASE="{output_dir}"
else
    OUTPUT_BASE="$PROJECT_DIR/{output_dir}"
fi

if [[ "{source_exp_dir}" = /* ]]; then
    SOURCE_EXP_DIR="{source_exp_dir}"
else
    SOURCE_EXP_DIR="$PROJECT_DIR/{source_exp_dir}"
fi

# Setup transfer experiment directory
TRANSFER_DIR="${{OUTPUT_BASE}}/{transfer_name}_${{SLURM_JOB_ID}}"
mkdir -p "${{TRANSFER_DIR}}/checkpoints"
mkdir -p "${{TRANSFER_DIR}}/logs"

# Source checkpoint path
SOURCE_CHECKPOINT="${{SOURCE_EXP_DIR}}/checkpoints/checkpoint_{checkpoint_timesteps}.zip"

echo "Transfer experiment directory: $TRANSFER_DIR"
echo "Source checkpoint: $SOURCE_CHECKPOINT"
echo "Source game: {source_game}"
echo "Target game: {target_game}"
echo "Checkpoint timesteps: {checkpoint_timesteps:,}"

# Wait for checkpoint to exist (with timeout)
echo "Waiting for checkpoint file..."
WAIT_COUNT=0
MAX_WAIT=1800  # 30 minutes (in 1-second intervals)

while [ ! -f "$SOURCE_CHECKPOINT" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "  Still waiting... ($((WAIT_COUNT / 60)) minutes elapsed)"
    fi
done

if [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found after waiting $((MAX_WAIT / 60)) minutes"
    echo "Expected: $SOURCE_CHECKPOINT"
    exit 1
fi

echo "Checkpoint found! Starting transfer learning..."

# Run transfer learning
"""

    freeze_flag = "--freeze-encoder" if freeze_encoder else ""
    reinit_flag = "--reinit-head" if reinit_head else ""

    script_content += f"""
python train_{algorithm}.py \\
    --game {target_game} \\
    --timesteps {total_timesteps} \\
    --checkpoint-dir "${{TRANSFER_DIR}}/checkpoints" \\
    --log-dir "${{TRANSFER_DIR}}/logs" \\
    --pretrained "$SOURCE_CHECKPOINT" \\
    --checkpoint-freq {checkpoint_timesteps // 10} \\
    --eval-freq {eval_freq} \\
    {freeze_flag} \\
    {reinit_flag}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Transfer learning completed successfully"
else
    echo "Transfer learning failed with exit code $EXIT_CODE"
fi

echo "Job completed: $(date)"
exit $EXIT_CODE
"""

    return script_content


def generate_phased_experiments_v2(
    source_games,
    target_games,
    algorithms,
    total_source_timesteps,
    num_checkpoints,
    target_timesteps,
    output_dir,
    eval_freq=10000,
    freeze_encoder=False,
    reinit_head=True,
    partition="gpu",
    source_time_limit="7-00:00:00",
    transfer_time_limit="2-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    venv_path=None,
):
    """
    Generate SLURM jobs for phased transfer learning (Version 2 - continuous source).

    Args:
        source_games: List of games to train source models on
        target_games: List of games to transfer to
        algorithms: List of algorithms to use
        total_source_timesteps: Total timesteps for source training (continuous)
        num_checkpoints: Number of checkpoints to save during source training
        target_timesteps: Timesteps for target training
        output_dir: Base output directory
        ... (other parameters)
    """
    os.makedirs(output_dir, exist_ok=True)
    scripts_dir = os.path.join(output_dir, "slurm_scripts_phased_v2")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "slurm_logs"), exist_ok=True)

    # Calculate checkpoint intervals
    checkpoint_interval = total_source_timesteps // num_checkpoints
    checkpoint_intervals = [checkpoint_interval * (i + 1) for i in range(num_checkpoints)]

    # Store submission commands
    submit_commands = []

    print(f"\n{'='*80}")
    print("PHASED TRANSFER LEARNING EXPERIMENT GENERATION (V2 - Continuous)")
    print(f"{'='*80}\n")
    print(f"Source games: {', '.join(source_games)}")
    print(f"Target games: {', '.join(target_games)}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Source training: {total_source_timesteps:,} timesteps (continuous)")
    print(f"Checkpoints at: {', '.join(f'{x:,}' for x in checkpoint_intervals)}")
    print(f"Target training: {target_timesteps:,} timesteps per transfer\n")

    # Generate scripts for each algorithm and source game
    for algorithm in algorithms:
        for source_game in source_games:
            print(f"\nGenerating jobs for {algorithm.upper()} on {source_game}:")
            print("-" * 60)

            source_exp_dir = f"{output_dir}/{algorithm}_{source_game}_source"

            # Generate ONE continuous source training job
            source_job_name = f"{algorithm}_{source_game}_source"
            source_script_path = os.path.join(scripts_dir, f"{source_job_name}.sh")

            source_script = create_continuous_source_slurm_script(
                job_name=source_job_name,
                algorithm=algorithm,
                game=source_game,
                total_timesteps=total_source_timesteps,
                checkpoint_intervals=checkpoint_intervals,
                output_dir=output_dir,
                eval_freq=eval_freq,
                partition=partition,
                time_limit=source_time_limit,
                mem=mem,
                cpus=cpus,
                gpus=gpus,
                venv_path=venv_path,
            )

            with open(source_script_path, "w") as f:
                f.write(source_script)
            os.chmod(source_script_path, 0o755)

            # Job ID variable for this source
            source_job_var = f"SOURCE_{algorithm.upper()}_{source_game.upper()}"

            # Add submission command for source
            submit_commands.append(
                f'{source_job_var}=$(sbatch --parsable {source_script_path})'
            )

            print(f"  Source: {source_job_name} (continuous, {total_source_timesteps:,} steps)")

            # Generate transfer jobs for each checkpoint to all target games
            for checkpoint_steps in checkpoint_intervals:
                for target_game in target_games:
                    if target_game == source_game:
                        continue  # Skip self-transfer

                    transfer_job_name = f"{algorithm}_{source_game}_to_{target_game}_ckpt{checkpoint_steps}"
                    transfer_script_path = os.path.join(scripts_dir, f"{transfer_job_name}.sh")

                    transfer_script = create_transfer_with_wait_slurm_script(
                        job_name=transfer_job_name,
                        algorithm=algorithm,
                        source_game=source_game,
                        target_game=target_game,
                        checkpoint_timesteps=checkpoint_steps,
                        total_timesteps=target_timesteps,
                        output_dir=output_dir,
                        source_exp_dir=source_exp_dir,
                        dependency_job_id_var=f"${{{source_job_var}}}",
                        eval_freq=eval_freq,
                        freeze_encoder=freeze_encoder,
                        reinit_head=reinit_head,
                        partition=partition,
                        time_limit=transfer_time_limit,
                        mem=mem,
                        cpus=cpus,
                        gpus=gpus,
                        venv_path=venv_path,
                    )

                    with open(transfer_script_path, "w") as f:
                        f.write(transfer_script)
                    os.chmod(transfer_script_path, 0o755)

                    # Add submission command
                    submit_commands.append(
                        f'sbatch --dependency=afterany:${{{source_job_var}}} {transfer_script_path} >/dev/null'
                    )

                    print(f"    → Transfer to {target_game} at {checkpoint_steps:,} steps")

    # Create master submission script
    submit_all_path = os.path.join(scripts_dir, "submit_all.sh")
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Master submission script for phased transfer learning (v2 - continuous)\n\n")
        f.write("set -e\n\n")

        for cmd in submit_commands:
            f.write(cmd + "\n")

        f.write(f"\necho 'Submitted {len(submit_commands)} jobs'\n")
        f.write("echo 'Source jobs will train continuously'\n")
        f.write("echo 'Transfer jobs will wait for checkpoints and start automatically'\n")
        f.write("echo 'Monitor with: squeue -u $USER'\n")

    os.chmod(submit_all_path, 0o755)

    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Total scripts: {len(submit_commands)}")
    print(f"\nTo submit all jobs:")
    print(f"  bash {submit_all_path}")
    print(f"\nHow it works:")
    print(f"  1. Source jobs train continuously (0 → {total_source_timesteps:,} steps)")
    print(f"  2. Checkpoints saved at: {', '.join(f'{x:,}' for x in checkpoint_intervals)}")
    print(f"  3. Transfer jobs depend on source but can start early")
    print(f"  4. Each transfer waits for its specific checkpoint file")
    print(f"  5. All jobs run with automatic SLURM dependencies")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate phased transfer learning jobs (v2 - continuous source)"
    )
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--source-games", type=str, nargs="+")
    parser.add_argument("--target-games", type=str, nargs="+")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       choices=["dqn", "ppo", "qrdqn", "sac"])
    parser.add_argument("--total-source-timesteps", type=int, default=None)
    parser.add_argument("--num-checkpoints", type=int, default=None,
                       help="Number of checkpoints to save (replaces num_phases)")
    parser.add_argument("--target-timesteps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results_phased_v2")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--reinit-head", action="store_true")
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--source-time-limit", type=str, default=None)
    parser.add_argument("--transfer-time-limit", type=str, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--venv-path", type=str, default=None)

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

        source_games = config.get("source_games", [])
        target_games = config.get("target_games", [])

        if "training" in config:
            training_config = config["training"]
            if args.total_source_timesteps is None:
                args.total_source_timesteps = training_config.get("total_source_timesteps", 50000000)
            if args.num_checkpoints is None:
                # Use num_phases from config as num_checkpoints
                args.num_checkpoints = training_config.get("num_phases", 5)
            if args.target_timesteps is None:
                args.target_timesteps = training_config.get("target_timesteps", 50000000)
            if args.eval_freq is None:
                args.eval_freq = training_config.get("eval_freq", 10000)
            if not args.freeze_encoder:
                args.freeze_encoder = training_config.get("freeze_encoder", False)
            if not args.reinit_head:
                args.reinit_head = training_config.get("reinit_head", True)

        if "slurm" in config:
            slurm_config = config["slurm"]
            if args.partition is None:
                args.partition = slurm_config.get("partition", "gpu")
            if args.source_time_limit is None:
                args.source_time_limit = slurm_config.get("source_time_limit", "7-00:00:00")
            if args.transfer_time_limit is None:
                args.transfer_time_limit = slurm_config.get("transfer_time_limit", "2-00:00:00")
            if args.mem is None:
                args.mem = slurm_config.get("mem", "32G")
            if args.cpus is None:
                args.cpus = slurm_config.get("cpus", 4)
            if args.gpus is None:
                args.gpus = slurm_config.get("gpus", 1)
            if args.venv_path is None:
                args.venv_path = slurm_config.get("venv_path")

        if "algorithms" in config and args.algorithms is None:
            args.algorithms = config["algorithms"]

        if args.output_dir == "results_phased_v2" and "output_dir" in config:
            args.output_dir = config["output_dir"] + "_v2"

    elif args.source_games and args.target_games:
        source_games = args.source_games
        target_games = args.target_games
    else:
        print("Error: Must provide --config or both --source-games and --target-games")
        exit(1)

    # Apply defaults
    if args.algorithms is None:
        args.algorithms = ["dqn", "ppo", "qrdqn"]
    if args.total_source_timesteps is None:
        args.total_source_timesteps = 50000000
    if args.num_checkpoints is None:
        args.num_checkpoints = 5
    if args.target_timesteps is None:
        args.target_timesteps = 50000000
    if args.eval_freq is None:
        args.eval_freq = 10000
    if args.partition is None:
        args.partition = "gpu"
    if args.source_time_limit is None:
        args.source_time_limit = "7-00:00:00"
    if args.transfer_time_limit is None:
        args.transfer_time_limit = "2-00:00:00"
    if args.mem is None:
        args.mem = "32G"
    if args.cpus is None:
        args.cpus = 4
    if args.gpus is None:
        args.gpus = 1

    if len(source_games) == 0 or len(target_games) == 0:
        print("Error: Must specify source and target games")
        exit(1)

    generate_phased_experiments_v2(
        source_games=source_games,
        target_games=target_games,
        algorithms=args.algorithms,
        total_source_timesteps=args.total_source_timesteps,
        num_checkpoints=args.num_checkpoints,
        target_timesteps=args.target_timesteps,
        output_dir=args.output_dir,
        eval_freq=args.eval_freq,
        freeze_encoder=args.freeze_encoder,
        reinit_head=args.reinit_head,
        partition=args.partition,
        source_time_limit=args.source_time_limit,
        transfer_time_limit=args.transfer_time_limit,
        mem=args.mem,
        cpus=args.cpus,
        gpus=args.gpus,
        venv_path=args.venv_path,
    )
