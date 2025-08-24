import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np

def train_ppo_agent():
    """Train PPO agent on Reacher-v5"""
    print("Training PPO agent on Reacher-v5...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environment
    env = make_vec_env('Reacher-v5', n_envs=4, seed=0)
    eval_env = Monitor(gym.make('Reacher-v5'))
    
    # Create callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=2000,
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0
    )
    
    # Train the model
    model.learn(
        total_timesteps=1000000,  # Adjust based on your needs
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/ppo_reacher_final")
    print("✅ Training completed! Model saved to models/ppo_reacher_final.zip")
    
    return model

def train_sac_agent():
    """Train SAC agent on Reacher-v5"""
    print("Training SAC agent on Reacher-v5...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environment
    env = gym.make('Reacher-v5')
    eval_env = Monitor(gym.make('Reacher-v5'))
    
    # Create callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Create SAC model
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1
    )
    
    # Train the model
    model.learn(
        total_timesteps=25000,  # SAC typically needs fewer steps
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/sac_reacher_final")
    print("✅ Training completed! Model saved to models/sac_reacher_final.zip")
    
    return model

if __name__ == "__main__":
    # Train PPO (comment out if you prefer SAC)
    # ppo_model = train_ppo_agent()
    
    # Train SAC (uncomment if you want to try SAC instead)
    sac_model = train_sac_agent()
