#!/usr/bin/env python3
"""
Generate SLURM job scripts for phased transfer learning experiments.

This script creates:
1. Source training jobs split into phases with dependencies
2. Transfer learning jobs that start after each checkpoint phase completes
3. Automatic dependency chaining for efficient resource usage

Each source task is trained in phases (e.g., 200M timesteps split into 10 phases of 20M each).
After each phase completes and creates a checkpoint, transfer jobs to all target games are started.
"""
import os
import json
import argparse
from pathlib import Path


def create_phased_source_slurm_script(
    job_name,
    algorithm,
    game,
    phase_id,
    phase_timesteps,
    output_dir,
    resume_from=None,
    dependency_job_id=None,
    checkpoint_freq=5000000,
    eval_freq=10000,
    partition="gpu",
    time_limit="1-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    venv_path=None,
):
    """Create a SLURM script for one phase of source training."""

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_logs/{job_name}_%j.out
#SBATCH --error={output_dir}/slurm_logs/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
"""

    # Add dependency if specified
    if dependency_job_id:
        script_content += f"#SBATCH --dependency=afterok:{dependency_job_id}\n"

    script_content += """
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Project directory (where scripts are located)
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

    # Determine absolute output directory
    exp_name = f"{algorithm}_{game}_source"

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
mkdir -p "${{EXP_DIR}}/signals"

echo "Experiment directory: $EXP_DIR"
echo "Phase: {phase_id}"
echo "Training for {phase_timesteps} timesteps"
"""

    # Build resume argument
    resume_arg = ""
    if resume_from:
        script_content += f"""
# Resume from previous phase checkpoint
RESUME_FROM="${{EXP_DIR}}/checkpoints/{resume_from}"
echo "Resuming from: $RESUME_FROM"
"""
        resume_arg = '--resume-from "$RESUME_FROM"'

    # Build signal file path
    signal_file = f"${{EXP_DIR}}/signals/{phase_id}.done"

    script_content += f"""
# Run phased training
python train_phased.py \\
    --algorithm {algorithm} \\
    --game {game} \\
    --phase-timesteps {phase_timesteps} \\
    --checkpoint-dir "${{EXP_DIR}}/checkpoints" \\
    --log-dir "${{EXP_DIR}}/logs" \\
    {resume_arg} \\
    --checkpoint-freq {checkpoint_freq} \\
    --eval-freq {eval_freq} \\
    --phase-id {phase_id} \\
    --signal-file "{signal_file}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Phase {phase_id} completed successfully"
    echo "Checkpoint saved and signal file created"
else
    echo "Phase {phase_id} failed with exit code $EXIT_CODE"
fi

echo "Job completed: $(date)"
exit $EXIT_CODE
"""

    return script_content


def create_transfer_slurm_script(
    job_name,
    algorithm,
    source_game,
    target_game,
    checkpoint_name,
    total_timesteps,
    output_dir,
    source_exp_dir,
    dependency_job_id,
    checkpoint_freq=5000000,
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
    """Create a SLURM script for transfer learning from a specific checkpoint."""

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_logs/{job_name}_%j.out
#SBATCH --error={output_dir}/slurm_logs/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --dependency=afterok:{dependency_job_id}

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

    transfer_name = f"{algorithm}_{source_game}_to_{target_game}_from_{checkpoint_name}"

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
SOURCE_CHECKPOINT="${{SOURCE_EXP_DIR}}/checkpoints/{checkpoint_name}.zip"

echo "Transfer experiment directory: $TRANSFER_DIR"
echo "Source checkpoint: $SOURCE_CHECKPOINT"
echo "Source game: {source_game}"
echo "Target game: {target_game}"

# Check that source checkpoint exists
if [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "ERROR: Source checkpoint not found: $SOURCE_CHECKPOINT"
    exit 1
fi

# Run transfer learning using appropriate training script
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
    --checkpoint-freq {checkpoint_freq} \\
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


def generate_phased_experiments(
    source_games,
    target_games,
    algorithms,
    total_source_timesteps,
    num_phases,
    target_timesteps,
    output_dir,
    checkpoint_freq=5000000,
    eval_freq=10000,
    freeze_encoder=False,
    reinit_head=True,
    partition="gpu",
    source_time_limit="1-00:00:00",
    transfer_time_limit="2-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    venv_path=None,
):
    """
    Generate all SLURM jobs for phased transfer learning experiments.

    Args:
        source_games: List of games to train source models on
        target_games: List of games to transfer to
        algorithms: List of algorithms to use
        total_source_timesteps: Total timesteps for source training
        num_phases: Number of phases to split source training into
        target_timesteps: Timesteps for target training
        output_dir: Base output directory
        ... (other SLURM parameters)
    """
    os.makedirs(output_dir, exist_ok=True)
    scripts_dir = os.path.join(output_dir, "slurm_scripts_phased")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "slurm_logs"), exist_ok=True)

    phase_timesteps = total_source_timesteps // num_phases

    # Track job dependencies
    # Key: (algorithm, source_game, phase_id) -> placeholder for job ID
    job_ids = {}

    # Store all job submission commands
    submit_commands = []

    print(f"\n{'='*80}")
    print("PHASED TRANSFER LEARNING EXPERIMENT GENERATION")
    print(f"{'='*80}\n")
    print(f"Source games: {', '.join(source_games)}")
    print(f"Target games: {', '.join(target_games)}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Total source training: {total_source_timesteps:,} timesteps")
    print(f"Split into {num_phases} phases of {phase_timesteps:,} timesteps each")
    print(f"Target training: {target_timesteps:,} timesteps per transfer\n")

    # Generate scripts for each algorithm and source game
    for algorithm in algorithms:
        for source_game in source_games:
            print(f"\nGenerating jobs for {algorithm.upper()} on {source_game}:")
            print("-" * 60)

            source_exp_dir = f"{output_dir}/{algorithm}_{source_game}_source"

            # Generate source training phases
            prev_job_var = None
            for phase in range(num_phases):
                phase_id = f"phase_{phase}"
                job_name = f"{algorithm}_{source_game}_source_{phase_id}"
                script_path = os.path.join(scripts_dir, f"{job_name}.sh")

                # Determine resume checkpoint
                resume_from = None if phase == 0 else f"phase_{phase-1}_model"

                # Create script (no job ID yet, will use variable)
                script_content = create_phased_source_slurm_script(
                    job_name=job_name,
                    algorithm=algorithm,
                    game=source_game,
                    phase_id=phase_id,
                    phase_timesteps=phase_timesteps,
                    output_dir=output_dir,
                    resume_from=resume_from,
                    dependency_job_id=f"${{{prev_job_var}}}" if prev_job_var else None,
                    checkpoint_freq=checkpoint_freq,
                    eval_freq=eval_freq,
                    partition=partition,
                    time_limit=source_time_limit,
                    mem=mem,
                    cpus=cpus,
                    gpus=gpus,
                    venv_path=venv_path,
                )

                with open(script_path, "w") as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)

                # Variable name for this job's ID
                job_var = f"SOURCE_{algorithm.upper()}_{source_game.upper()}_PHASE{phase}"
                job_ids[(algorithm, source_game, phase_id)] = job_var

                # Add submission command
                if prev_job_var:
                    submit_commands.append(
                        f'{job_var}=$(sbatch --parsable --dependency=afterok:${{{prev_job_var}}} {script_path})'
                    )
                else:
                    submit_commands.append(
                        f'{job_var}=$(sbatch --parsable {script_path})'
                    )

                print(f"  Phase {phase}: {job_name}")

                # Generate transfer jobs for this checkpoint to all target games
                checkpoint_name = phase_id + "_model"
                for target_game in target_games:
                    if target_game == source_game:
                        continue  # Skip self-transfer

                    transfer_job_name = f"{algorithm}_{source_game}_to_{target_game}_{phase_id}"
                    transfer_script_path = os.path.join(scripts_dir, f"{transfer_job_name}.sh")

                    transfer_script = create_transfer_slurm_script(
                        job_name=transfer_job_name,
                        algorithm=algorithm,
                        source_game=source_game,
                        target_game=target_game,
                        checkpoint_name=checkpoint_name,
                        total_timesteps=target_timesteps,
                        output_dir=output_dir,
                        source_exp_dir=source_exp_dir,
                        dependency_job_id=f"${{{job_var}}}",
                        checkpoint_freq=checkpoint_freq,
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

                    # Add submission command for transfer job
                    submit_commands.append(
                        f'TRANSFER_{job_var}_{target_game.upper()}=$(sbatch --parsable --dependency=afterok:${{{job_var}}} {transfer_script_path})'
                    )

                    print(f"    → Transfer to {target_game}: {transfer_job_name}")

                prev_job_var = job_var

    # Create master submission script
    submit_all_path = os.path.join(scripts_dir, "submit_all.sh")
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Master submission script for phased transfer learning experiments\n")
        f.write("# This script submits all jobs with proper dependency chaining\n\n")
        f.write("set -e  # Exit on error\n\n")

        for cmd in submit_commands:
            f.write(cmd + "\n")

        f.write(f"\necho 'Submitted {len(submit_commands)} jobs with dependency chaining'\n")
        f.write("echo 'Monitor with: squeue -u $USER'\n")

    os.chmod(submit_all_path, 0o755)

    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Total scripts generated: {len(submit_commands)}")
    print(f"\nTo submit all jobs with dependency chaining:")
    print(f"  bash {submit_all_path}")
    print(f"\nTo monitor jobs:")
    print(f"  squeue -u $USER")
    print(f"  watch -n 5 'squeue -u $USER'")
    print(f"\nExperiment structure:")
    print(f"  - Each source game trains in {num_phases} phases")
    print(f"  - After each phase completes, transfer jobs start for all target games")
    print(f"  - Transfer jobs train for {target_timesteps:,} timesteps")
    print(f"  - All jobs use automatic SLURM dependency chaining")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate phased transfer learning SLURM jobs"
    )
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--source-games", type=str, nargs="+",
                       help="List of source games")
    parser.add_argument("--target-games", type=str, nargs="+",
                       help="List of target games")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       choices=["dqn", "ppo", "qrdqn", "sac"],
                       help="RL algorithms to use")
    parser.add_argument("--total-source-timesteps", type=int, default=None,
                       help="Total timesteps for source training")
    parser.add_argument("--num-phases", type=int, default=None,
                       help="Number of phases to split source training into")
    parser.add_argument("--target-timesteps", type=int, default=None,
                       help="Timesteps for target training")
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                       help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=None,
                       help="Evaluation frequency")
    parser.add_argument("--output-dir", type=str, default="results_phased",
                       help="Output directory for experiments")
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="Freeze encoder during transfer")
    parser.add_argument("--reinit-head", action="store_true",
                       help="Reinitialize head during transfer")
    parser.add_argument("--partition", type=str, default=None,
                       help="SLURM partition")
    parser.add_argument("--source-time-limit", type=str, default=None,
                       help="Time limit per source phase")
    parser.add_argument("--transfer-time-limit", type=str, default=None,
                       help="Time limit per transfer job")
    parser.add_argument("--mem", type=str, default=None,
                       help="Memory per job")
    parser.add_argument("--cpus", type=int, default=None,
                       help="CPUs per task")
    parser.add_argument("--gpus", type=int, default=None,
                       help="GPUs per task")
    parser.add_argument("--venv-path", type=str, default=None,
                       help="Path to virtual environment")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

        source_games = config.get("source_games", [])
        target_games = config.get("target_games", [])

        if "training" in config:
            training_config = config["training"]
            if args.total_source_timesteps is None:
                args.total_source_timesteps = training_config.get("total_source_timesteps", 200000000)
            if args.num_phases is None:
                args.num_phases = training_config.get("num_phases", 10)
            if args.target_timesteps is None:
                args.target_timesteps = training_config.get("target_timesteps", 200000000)
            if args.checkpoint_freq is None:
                args.checkpoint_freq = training_config.get("checkpoint_freq", 5000000)
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
                args.source_time_limit = slurm_config.get("source_time_limit", "1-00:00:00")
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

        if args.output_dir == "results_phased" and "output_dir" in config:
            args.output_dir = config["output_dir"]

    elif args.source_games and args.target_games:
        source_games = args.source_games
        target_games = args.target_games
    else:
        print("Error: Must provide either --config or both --source-games and --target-games")
        exit(1)

    # Apply defaults
    if args.algorithms is None:
        args.algorithms = ["dqn", "ppo", "qrdqn"]
    if args.total_source_timesteps is None:
        args.total_source_timesteps = 200000000
    if args.num_phases is None:
        args.num_phases = 10
    if args.target_timesteps is None:
        args.target_timesteps = 200000000
    if args.checkpoint_freq is None:
        args.checkpoint_freq = 5000000
    if args.eval_freq is None:
        args.eval_freq = 10000
    if args.partition is None:
        args.partition = "gpu"
    if args.source_time_limit is None:
        args.source_time_limit = "1-00:00:00"
    if args.transfer_time_limit is None:
        args.transfer_time_limit = "2-00:00:00"
    if args.mem is None:
        args.mem = "32G"
    if args.cpus is None:
        args.cpus = 4
    if args.gpus is None:
        args.gpus = 1

    if len(source_games) == 0 or len(target_games) == 0:
        print("Error: Must specify at least one source and target game")
        exit(1)

    generate_phased_experiments(
        source_games=source_games,
        target_games=target_games,
        algorithms=args.algorithms,
        total_source_timesteps=args.total_source_timesteps,
        num_phases=args.num_phases,
        target_timesteps=args.target_timesteps,
        output_dir=args.output_dir,
        checkpoint_freq=args.checkpoint_freq,
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
