# Atari Transfer Learning with DQN and PPO

A simple, clean codebase for training deep reinforcement learning models (DQN and PPO) on Atari games with transfer learning support. Built on Stable-Baselines3 with SLURM cluster integration for parallel experiments.

## Features

- DQN and PPO implementations using Stable-Baselines3
- Transfer learning between Atari game pairs
- Automatic checkpointing and comprehensive logging
- SLURM job generation for parallel cluster execution
- TensorBoard integration for monitoring
- Minimal dependencies and straightforward interfaces

## Installation

```bash
cd atari-transfer-rl
pip install -r requirements.txt
```

For conda users:
```bash
conda create -n atari-rl python=3.10
conda activate atari-rl
pip install -r requirements.txt
```

## Project Structure

```
atari-transfer-rl/
├── train_dqn.py              # DQN training script
├── train_ppo.py              # PPO training script
├── transfer_learning.py      # Transfer learning wrapper
├── generate_slurm_jobs.py    # SLURM job generator
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── checkpoints/              # Model checkpoints (auto-created)
├── logs/                     # Training logs (auto-created)
└── results/                  # Experiment results (auto-created)
```

## Usage

### 1. Single Game Training

Train DQN on a single game:
```bash
python train_dqn.py --game Pong --timesteps 1000000
```

Train PPO on a single game:
```bash
python train_ppo.py --game Breakout --timesteps 1000000
```

### 2. Transfer Learning

Train on a source game, then transfer to a target game:
```bash
python transfer_learning.py \
    --algorithm dqn \
    --source-game Pong \
    --target-game Breakout \
    --source-timesteps 1000000 \
    --target-timesteps 1000000
```

### 3. Batch Experiments with SLURM

Edit [config.json](config.json) to specify your games list:
```json
{
  "games": ["Pong", "Breakout", "SpaceInvaders", "Qbert"]
}
```

Generate SLURM job scripts for all pairwise combinations:
```bash
python generate_slurm_jobs.py --config config.json
```

Or specify games directly:
```bash
python generate_slurm_jobs.py \
    --games Pong Breakout SpaceInvaders \
    --algorithms dqn ppo \
    --partition gpu \
    --time-limit 24:00:00 \
    --mem 32G \
    --cpus 4 \
    --gpus 1 \
    --conda-env atari-rl
```

Submit all jobs:
```bash
bash results/slurm_scripts/submit_all.sh
```

Or submit individual jobs:
```bash
sbatch results/slurm_scripts/dqn_Pong_to_Breakout.sh
```

## Configuration

### Training Parameters

DQN parameters (train_dqn.py):
- `--lr`: Learning rate (default: 1e-4)
- `--buffer-size`: Replay buffer size (default: 100000)
- `--checkpoint-freq`: Steps between checkpoints (default: 50000)
- `--eval-freq`: Steps between evaluations (default: 10000)

PPO parameters (train_ppo.py):
- `--lr`: Learning rate (default: 2.5e-4)
- `--n-steps`: Steps per update (default: 128)
- `--batch-size`: Batch size (default: 256)

### SLURM Configuration

Adjust SLURM parameters in [config.json](config.json) or via command-line:
- `partition`: SLURM partition name
- `time_limit`: Job time limit (HH:MM:SS)
- `mem`: Memory allocation
- `cpus`: CPU cores per task
- `gpus`: GPUs per task
- `conda_env`: Conda environment to activate

## Monitoring

### TensorBoard

Monitor training in real-time:
```bash
tensorboard --logdir results/
```

### Training Logs

CSV logs are saved in the experiment directories:
```
results/
└── dqn_Pong_to_Breakout_20260113_123456/
    ├── config.json
    ├── source_logs/
    │   └── progress.csv
    ├── target_logs/
    │   └── progress.csv
    └── results.json
```

### Checkpoints

Models are automatically saved at:
- Regular intervals (every `checkpoint_freq` steps)
- Best performing model (based on evaluation)
- Final model at end of training

## Output Structure

Each transfer learning experiment creates:
```
results/
└── <algorithm>_<source>_to_<target>_<timestamp>/
    ├── config.json                    # Experiment configuration
    ├── results.json                   # Paths to models and logs
    ├── source_checkpoints/            # Source game checkpoints
    │   ├── final_model.zip
    │   └── best_model.zip
    ├── source_logs/                   # Source game logs
    ├── target_checkpoints/            # Target game checkpoints
    └── target_logs/                   # Target game logs
```

## Available Atari Games

Common games (add "-v5" suffix when using ALE):
- Pong
- Breakout
- SpaceInvaders
- Qbert
- MsPacman
- Seaquest
- BeamRider
- Enduro
- Asterix
- Assault

Full list: https://gymnasium.farama.org/environments/atari/

## Example Workflow

1. Setup environment:
```bash
conda create -n atari-rl python=3.10
conda activate atari-rl
pip install -r requirements.txt
```

2. Test single training run:
```bash
python train_dqn.py --game Pong --timesteps 100000
```

3. Edit [config.json](config.json) with your game list

4. Generate SLURM jobs:
```bash
python generate_slurm_jobs.py --config config.json
```

5. Submit to cluster:
```bash
bash results/slurm_scripts/submit_all.sh
```

6. Monitor progress:
```bash
# Check SLURM jobs
squeue -u $USER

# Monitor with TensorBoard
tensorboard --logdir results/
```

## Notes

- Each pairwise combination is run in both directions (A→B and B→A)
- For N games, this generates 2 × N × (N-1) experiments per algorithm
- Default: 6 games × 2 algorithms = 60 total experiments
- Checkpoint frequency balances storage and recovery capability
- Evaluation runs every `eval_freq` steps with 5 episodes

## Troubleshooting

**Out of memory errors**: Reduce `--buffer-size` for DQN or `--batch-size` for PPO

**Slow training**: Increase `--cpus` or use multiple environments (modify scripts to use `SubprocVecEnv`)

**Missing ROMs**: The `accept-rom-license` flag in requirements.txt auto-downloads ROMs. If issues persist, manually install: `pip install "gymnasium[accept-rom-license]"`

**SLURM job failures**: Check `results/slurm_logs/` for error messages
