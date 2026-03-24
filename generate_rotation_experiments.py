#!/usr/bin/env python3
"""
Generate SLURM job scripts for rotation-based transfer learning experiments.

Experiment design:
  - Source task: SpaceInvaders (10M steps)
  - Target task: Pong
  - Two rotation variants:
      A) source_rotated: train SpaceInvaders with 90° rotation, transfer to Pong (no rotation)
      B) target_rotated: train SpaceInvaders (no rotation), transfer to Pong with 90° rotation
  - Algorithms: DQN, PPO
  - Seeds: configurable (default 3)

Each seed gets its own independent source and transfer job pair.
"""
import os
import json
import argparse


def create_source_script(
    job_name,
    algorithm,
    game,
    total_timesteps,
    output_dir,
    rotate=False,
    eval_freq=10000,
    partition="gpu",
    time_limit="2-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    venv_path=None,
    seed=None,
):
    rotate_flag = "--rotate" if rotate else ""
    seed_flag = f"--seed {seed}" if seed is not None else ""
    exp_name = job_name

    script = f"""#!/bin/bash
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

PROJECT_DIR=$SLURM_SUBMIT_DIR
cd $PROJECT_DIR
"""

    if venv_path:
        script += f"""
if [ -f "{venv_path}/bin/activate" ]; then
    source {venv_path}/bin/activate
else
    echo "ERROR: venv not found at {venv_path}"
    exit 1
fi
"""

    script += f"""
if [[ "{output_dir}" = /* ]]; then
    OUTPUT_BASE="{output_dir}"
else
    OUTPUT_BASE="$PROJECT_DIR/{output_dir}"
fi

EXP_DIR="${{OUTPUT_BASE}}/{exp_name}"
mkdir -p "${{EXP_DIR}}/checkpoints"
mkdir -p "${{EXP_DIR}}/logs"

echo "Training {algorithm.upper()} on {game} (rotate={rotate}) for {total_timesteps:,} steps"

python train_continuous_checkpoints.py \\
    --algorithm {algorithm} \\
    --game {game} \\
    --timesteps {total_timesteps} \\
    --checkpoint-dir "${{EXP_DIR}}/checkpoints" \\
    --log-dir "${{EXP_DIR}}/logs" \\
    --checkpoint-intervals {total_timesteps} \\
    --eval-freq {eval_freq} \\
    {rotate_flag} \\
    {seed_flag}

EXIT_CODE=$?
echo "Job completed: $(date)"
exit $EXIT_CODE
"""
    return script


def create_transfer_script(
    job_name,
    algorithm,
    source_job_name,
    source_exp_dir,
    target_game,
    source_timesteps,
    target_timesteps,
    output_dir,
    dependency_var,
    rotate=False,
    freeze_encoder=False,
    reinit_head=True,
    eval_freq=10000,
    partition="gpu",
    time_limit="2-00:00:00",
    mem="32G",
    cpus=4,
    gpus=1,
    venv_path=None,
    seed=None,
):
    rotate_flag = "--rotate" if rotate else ""
    freeze_flag = "--freeze-encoder" if freeze_encoder else ""
    reinit_flag = "--reinit-head" if reinit_head else ""
    seed_flag = f"--seed {seed}" if seed is not None else ""
    checkpoint_freq = max(source_timesteps // 10, 50000)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_logs/{job_name}_%j.out
#SBATCH --error={output_dir}/slurm_logs/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --dependency=after:{dependency_var}

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

PROJECT_DIR=$SLURM_SUBMIT_DIR
cd $PROJECT_DIR
"""

    if venv_path:
        script += f"""
if [ -f "{venv_path}/bin/activate" ]; then
    source {venv_path}/bin/activate
else
    echo "ERROR: venv not found at {venv_path}"
    exit 1
fi
"""

    script += f"""
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

TRANSFER_DIR="${{OUTPUT_BASE}}/{job_name}"
mkdir -p "${{TRANSFER_DIR}}/checkpoints"
mkdir -p "${{TRANSFER_DIR}}/logs"

SOURCE_CHECKPOINT="${{SOURCE_EXP_DIR}}/checkpoints/checkpoint_{source_timesteps}.zip"

echo "Transfer: source checkpoint=$SOURCE_CHECKPOINT"
echo "Target game: {target_game} (rotate={rotate})"

# Wait up to 30 minutes for checkpoint (should already exist when this job starts)
WAIT_COUNT=0
MAX_WAIT=1800
while [ ! -f "$SOURCE_CHECKPOINT" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "  Waiting for checkpoint... ($((WAIT_COUNT / 60)) min elapsed)"
    fi
done

if [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found after $((MAX_WAIT / 60)) minutes: $SOURCE_CHECKPOINT"
    exit 1
fi

echo "Checkpoint found. Starting transfer learning..."

python train_{algorithm}.py \\
    --game {target_game} \\
    --timesteps {target_timesteps} \\
    --checkpoint-dir "${{TRANSFER_DIR}}/checkpoints" \\
    --log-dir "${{TRANSFER_DIR}}/logs" \\
    --pretrained "$SOURCE_CHECKPOINT" \\
    --checkpoint-freq {checkpoint_freq} \\
    --eval-freq {eval_freq} \\
    {freeze_flag} \\
    {reinit_flag} \\
    {rotate_flag} \\
    {seed_flag}

EXIT_CODE=$?
echo "Job completed: $(date)"
exit $EXIT_CODE
"""
    return script


def generate_rotation_experiments(config):
    algorithms = config["algorithms"]
    source_game = config["source_game"]
    target_game = config["target_game"]
    source_timesteps = config["training"]["source_timesteps"]
    target_timesteps = config["training"]["target_timesteps"]
    eval_freq = config["training"].get("eval_freq", 10000)
    freeze_encoder = config["training"].get("freeze_encoder", False)
    reinit_head = config["training"].get("reinit_head", True)
    seeds = config["training"]["seeds"]
    output_dir = config["output_dir"]

    slurm = config["slurm"]
    partition = slurm.get("partition", "gpu")
    source_time_limit = slurm.get("source_time_limit", "1-00:00:00")
    transfer_time_limit = slurm.get("transfer_time_limit", "1-00:00:00")
    mem = slurm.get("mem", "32G")
    cpus = slurm.get("cpus", 4)
    gpus = slurm.get("gpus", 1)
    venv_path = slurm.get("venv_path")

    scripts_dir = os.path.join(output_dir, "slurm_scripts_rotation")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "slurm_logs"), exist_ok=True)

    # Rotation variants: (source_rotated, target_rotated, variant_label)
    rotation_variants = [
        (True,  False, "srcrot"),  # Variant A: rotated source -> normal target
        (False, True,  "tgtrot"),  # Variant B: normal source -> rotated target
    ]

    submit_commands = []

    print(f"\n{'='*70}")
    print("ROTATION TRANSFER EXPERIMENT GENERATION")
    print(f"{'='*70}")
    print(f"Source: {source_game} ({source_timesteps:,} steps)")
    print(f"Target: {target_game} ({target_timesteps:,} steps)")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Seeds: {seeds}")
    print(f"Variants: source_rotated->target, source->target_rotated")
    print(f"Output: {output_dir}\n")

    total_jobs = 0

    for algorithm in algorithms:
        for src_rot, tgt_rot, variant in rotation_variants:
            for seed in seeds:
                src_label = f"{source_game}_rot" if src_rot else source_game
                tgt_label = f"{target_game}_rot" if tgt_rot else target_game
                seed_label = f"s{seed}"

                source_job = f"{algorithm}_{src_label}_{seed_label}"
                transfer_job = f"{algorithm}_{src_label}_to_{tgt_label}_{seed_label}"

                source_exp_dir = f"{output_dir}/{source_job}"

                # Source script
                source_script = create_source_script(
                    job_name=source_job,
                    algorithm=algorithm,
                    game=source_game,
                    total_timesteps=source_timesteps,
                    output_dir=output_dir,
                    rotate=src_rot,
                    eval_freq=eval_freq,
                    partition=partition,
                    time_limit=source_time_limit,
                    mem=mem,
                    cpus=cpus,
                    gpus=gpus,
                    venv_path=venv_path,
                    seed=seed,
                )

                source_path = os.path.join(scripts_dir, f"{source_job}.sh")
                with open(source_path, "w") as f:
                    f.write(source_script)
                os.chmod(source_path, 0o755)

                # Job ID variable (safe for bash)
                job_var = f"JOB_{algorithm.upper()}_{variant.upper()}_{seed}"
                submit_commands.append(
                    f"{job_var}=$(sbatch --parsable {source_path})"
                )

                # Transfer script
                transfer_script = create_transfer_script(
                    job_name=transfer_job,
                    algorithm=algorithm,
                    source_job_name=source_job,
                    source_exp_dir=source_exp_dir,
                    target_game=target_game,
                    source_timesteps=source_timesteps,
                    target_timesteps=target_timesteps,
                    output_dir=output_dir,
                    dependency_var=f"${{{job_var}}}",
                    rotate=tgt_rot,
                    freeze_encoder=freeze_encoder,
                    reinit_head=reinit_head,
                    eval_freq=eval_freq,
                    partition=partition,
                    time_limit=transfer_time_limit,
                    mem=mem,
                    cpus=cpus,
                    gpus=gpus,
                    venv_path=venv_path,
                    seed=seed,
                )

                transfer_path = os.path.join(scripts_dir, f"{transfer_job}.sh")
                with open(transfer_path, "w") as f:
                    f.write(transfer_script)
                os.chmod(transfer_path, 0o755)

                submit_commands.append(
                    f"sbatch --dependency=after:${{{job_var}}} {transfer_path} >/dev/null"
                )

                rot_desc = "src_rot->tgt" if src_rot else "src->tgt_rot"
                print(f"  [{algorithm.upper()}] {rot_desc} seed={seed}: {source_job} -> {transfer_job}")
                total_jobs += 2

    # Master submit script
    submit_all_path = os.path.join(scripts_dir, "submit_all.sh")
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all rotation transfer learning experiments\n\n")
        f.write("set -e\n\n")
        for cmd in submit_commands:
            f.write(cmd + "\n")
        f.write(f"\necho 'Submitted {total_jobs} jobs ({total_jobs // 2} source + {total_jobs // 2} transfer)'\n")
        f.write("echo 'Monitor with: squeue -u $USER'\n")
    os.chmod(submit_all_path, 0o755)

    print(f"\n{'='*70}")
    print(f"Generated {total_jobs} scripts ({total_jobs // 2} source + {total_jobs // 2} transfer)")
    print(f"Scripts: {scripts_dir}")
    print(f"\nTo submit all jobs:")
    print(f"  bash {submit_all_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate rotation transfer learning SLURM jobs"
    )
    parser.add_argument("--config", type=str, default="config_rotation.json",
                        help="JSON config file (default: config_rotation.json)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    generate_rotation_experiments(config)
