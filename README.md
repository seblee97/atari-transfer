# Atari Transfer Learning with DQN and PPO

A simple, clean codebase for training deep reinforcement learning models (DQN and PPO) on Atari games with transfer learning support. Built on Stable-Baselines3 with SLURM cluster integration for parallel experiments.

## Features

- Multiple RL algorithms: DQN, PPO, SAC, and QR-DQN implementations using Stable-Baselines3
- Transfer learning between Atari game pairs with freeze encoder + reinitialize head support
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
├── train_sac.py              # SAC training script
├── train_qrdqn.py            # QR-DQN training script
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

Train SAC on a single game:
```bash
python train_sac.py --game SpaceInvaders --timesteps 1000000
```

Train QR-DQN on a single game:
```bash
python train_qrdqn.py --game Tennis --timesteps 1000000
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

**NEW: Freeze Encoder Transfer Learning**

Train with frozen encoder and reinitialized head (recommended for better transfer):
```bash
python transfer_learning.py \
    --algorithm dqn \
    --source-game Pong \
    --target-game Breakout \
    --source-timesteps 1000000 \
    --target-timesteps 1000000 \
    --freeze-encoder \
    --reinit-head
```

This approach:
- Freezes the CNN encoder layers trained on the source game
- Reinitializes the policy/value head with fresh weights
- Only trains the head on the target game
- Often leads to better transfer performance and faster convergence

### 3. Batch Experiments with SLURM

Edit [config.json](config.json) to specify your games list and transfer settings:
```json
{
  "games": ["Pong", "Breakout", "SpaceInvaders", "Tennis"],
  "algorithms": ["dqn", "ppo", "sac", "qrdqn"],
  "training": {
    "source_timesteps": 1000000,
    "target_timesteps": 1000000,
    "freeze_encoder": true,
    "reinit_head": true
  }
}
```

Generate SLURM job scripts for all pairwise combinations:
```bash
python generate_slurm_jobs.py --config config.json
```

Or specify games directly:
```bash
python generate_slurm_jobs.py \
    --games Pong Breakout SpaceInvaders Tennis \
    --algorithms dqn ppo sac qrdqn \
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
- `--freeze-encoder`: Freeze CNN encoder layers during transfer (flag)
- `--reinit-head`: Reinitialize final layer weights before transfer (flag)

PPO parameters (train_ppo.py):
- `--lr`: Learning rate (default: 2.5e-4)
- `--n-steps`: Steps per update (default: 128)
- `--batch-size`: Batch size (default: 256)
- `--freeze-encoder`: Freeze CNN encoder layers during transfer (flag)
- `--reinit-head`: Reinitialize policy/value heads before transfer (flag)

SAC parameters (train_sac.py):
- `--lr`: Learning rate (default: 3e-4)
- `--buffer-size`: Replay buffer size (default: 100000)
- `--tau`: Target network update rate (default: 0.005)
- `--gamma`: Discount factor (default: 0.99)
- `--freeze-encoder`: Freeze CNN encoder layers during transfer (flag)
- `--reinit-head`: Reinitialize actor/critic heads before transfer (flag)

QR-DQN parameters (train_qrdqn.py):
- `--lr`: Learning rate (default: 1e-4)
- `--buffer-size`: Replay buffer size (default: 100000)
- `--n-quantiles`: Number of quantiles for distributional RL (default: 200)
- `--freeze-encoder`: Freeze CNN encoder layers during transfer (flag)
- `--reinit-head`: Reinitialize quantile head before transfer (flag)

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

## Algorithm Comparison

**DQN (Deep Q-Network)**
- Off-policy value-based algorithm
- Uses experience replay and target networks
- Good for discrete action spaces
- Memory efficient with replay buffer

**PPO (Proximal Policy Optimization)**
- On-policy policy gradient algorithm
- Clipped surrogate objective for stable updates
- Generally more sample efficient than vanilla policy gradient
- Good balance of performance and stability

**SAC (Soft Actor-Critic)**
- Off-policy actor-critic algorithm
- Maximum entropy framework for exploration
- Supports both continuous and discrete action spaces
- Robust to hyperparameters
- Note: Standard SAC is designed for continuous actions; for Atari you may want to use discrete action variants or stick with DQN/PPO/QR-DQN

**QR-DQN (Quantile Regression DQN)**
- Distributional RL extension of DQN
- Models full distribution of returns instead of just expectation
- Uses quantile regression for better value estimation
- Often outperforms standard DQN

## Notes

- Each pairwise combination is run in both directions (A→B and B→A)
- For N games, this generates 2 × N × (N-1) experiments per algorithm
- Default: 4 games × 4 algorithms = 96 total experiments
- Checkpoint frequency balances storage and recovery capability
- Evaluation runs every `eval_freq` steps with 5 episodes

## Troubleshooting

**Out of memory errors**: Reduce `--buffer-size` for DQN or `--batch-size` for PPO

**Slow training**: Increase `--cpus` or use multiple environments (modify scripts to use `SubprocVecEnv`)

**Missing ROMs**: The `accept-rom-license` flag in requirements.txt auto-downloads ROMs. If issues persist, manually install: `pip install "gymnasium[accept-rom-license]"`

**SLURM job failures**: Check `results/slurm_logs/` for error messages

**SAC with discrete actions**: SAC is designed for continuous action spaces. For Atari (discrete actions), DQN, PPO, or QR-DQN are recommended. The SAC implementation is provided for completeness but may require modifications for optimal performance on discrete action environments.

**QR-DQN requirements**: QR-DQN requires `sb3-contrib` package, which is included in requirements.txt
