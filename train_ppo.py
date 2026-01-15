#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import gymnasium as gym
import ale_py
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file

# Register ALE environments
gym.register_envs(ale_py)

def make_atari_env(game_name):
    def _init():
        env = gym.make(f"ALE/{game_name}-v5", render_mode=None)
        env = AtariWrapper(env)
        return env
    return _init

def train_ppo(
    game_name,
    total_timesteps,
    checkpoint_dir,
    log_dir,
    pretrained_model=None,
    checkpoint_freq=50000,
    eval_freq=10000,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    clip_range=0.1,
    freeze_encoder=False,
    reinit_head=False,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_atari_env(game_name)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_atari_env(game_name)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    if pretrained_model and os.path.exists(pretrained_model):
        print(f"Loading pretrained model from {pretrained_model}")

        try:
            # Try to load the model directly
            # Use custom_objects to handle potential incompatibilities with zoo models
            custom_objects = {}
            model = PPO.load(pretrained_model, env=env, custom_objects=custom_objects)
            model.set_logger(logger)

            # Handle freeze encoder and reinit head for same action space
            if freeze_encoder or reinit_head:
                policy = model.policy

                if reinit_head:
                    print("Reinitializing policy and value heads")
                    # PPO has separate action_net and value_net heads
                    if hasattr(policy, 'action_net'):
                        torch.nn.init.orthogonal_(policy.action_net.weight, gain=0.01)
                        torch.nn.init.constant_(policy.action_net.bias, 0.0)
                    if hasattr(policy, 'value_net'):
                        torch.nn.init.orthogonal_(policy.value_net.weight, gain=1)
                        torch.nn.init.constant_(policy.value_net.bias, 0.0)

                if freeze_encoder:
                    print("Freezing CNN encoder layers")
                    # Freeze the shared feature extractor (CNN)
                    if hasattr(policy, 'features_extractor'):
                        for param in policy.features_extractor.parameters():
                            param.requires_grad = False

                        # Also freeze the MLP extractor's shared layers if present
                        if hasattr(policy, 'mlp_extractor'):
                            if hasattr(policy.mlp_extractor, 'shared_net'):
                                for param in policy.mlp_extractor.shared_net.parameters():
                                    param.requires_grad = False

                    print(f"Trainable parameters: {sum(p.numel() for p in model.policy.parameters() if p.requires_grad)}")
                    print(f"Frozen parameters: {sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)}")

        except ValueError as e:
            if "Action spaces do not match" in str(e):
                print(f"Action space mismatch detected: {e}")
                print("Transferring encoder weights only to new model with correct action space")

                # Load the saved data
                _, _, pytorch_variables = load_from_zip_file(pretrained_model)

                # Create a new model with the correct action space
                model = PPO(
                    "CnnPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    clip_range=clip_range,
                    verbose=1,
                    tensorboard_log=log_dir,
                )
                model.set_logger(logger)

                # Transfer encoder weights from pretrained model
                if pytorch_variables is not None and 'policy' in pytorch_variables:
                    pretrained_state = pytorch_variables['policy']
                    current_state = model.policy.state_dict()

                    # Transfer only the CNN feature extractor weights
                    transferred_keys = []
                    for key in current_state.keys():
                        if 'features_extractor' in key and key in pretrained_state:
                            if current_state[key].shape == pretrained_state[key].shape:
                                current_state[key] = pretrained_state[key]
                                transferred_keys.append(key)

                    model.policy.load_state_dict(current_state)
                    print(f"Transferred {len(transferred_keys)} encoder weight tensors")

                # Freeze encoder if requested
                if freeze_encoder:
                    print("Freezing transferred CNN encoder layers")
                    if hasattr(model.policy, 'features_extractor'):
                        for param in model.policy.features_extractor.parameters():
                            param.requires_grad = False

                    print(f"Trainable parameters: {sum(p.numel() for p in model.policy.parameters() if p.requires_grad)}")
                    print(f"Frozen parameters: {sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)}")
            else:
                raise
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            verbose=1,
            tensorboard_log=log_dir,
        )
        model.set_logger(logger)

    print(f"Training PPO on {game_name} for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
    )

    final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    env.close()
    eval_env.close()

    return final_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Atari game")
    parser.add_argument("--game", type=str, required=True, help="Atari game name (e.g., Pong, Breakout)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=128, help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze CNN encoder layers during transfer")
    parser.add_argument("--reinit-head", action="store_true", help="Reinitialize the policy and value head weights")

    args = parser.parse_args()

    train_ppo(
        game_name=args.game,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        pretrained_model=args.pretrained,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        freeze_encoder=args.freeze_encoder,
        reinit_head=args.reinit_head,
    )
