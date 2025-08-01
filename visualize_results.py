import gymnasium as gym
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_training_curves():
    """Plot training reward curves from logs"""
    if not os.path.exists('logs/evaluations.npz'):
        print("No evaluation logs found. Run training first.")
        return
    
    # Load evaluation data
    data = np.load('logs/evaluations.npz')
    timesteps = data['timesteps']
    results = data['results']
    
    # Calculate mean and std
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label='Mean Reward', linewidth=2)
    plt.fill_between(timesteps, 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     alpha=0.3, label='±1 std')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Training Progress - Reacher-v4')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final mean reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")

def evaluate_trained_model(model_path="models/ppo_reacher_final.zip", render=True):
    """Evaluate and visualize trained model"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "sac" in model_path.lower():
        model = SAC.load(model_path)
    else:
        print("Unknown model type. Assuming PPO.")
        model = PPO.load(model_path)
    
    # Create environment
    env = gym.make('Reacher-v4', render_mode='human' if render else None)
    
    # Evaluate model
    n_episodes = 5
    episode_rewards = []
    
    print(f"Evaluating model for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            if step > 200:  # Prevent infinite episodes
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    env.close()
    
    # Print statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nEvaluation Results:")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    
    return episode_rewards

def compare_random_vs_trained():
    """Compare random policy vs trained policy"""
    model_path = "models/ppo_reacher_final.zip"
    
    if not os.path.exists(model_path):
        print("Trained model not found. Train first.")
        return
    
    # Load trained model
    model = PPO.load(model_path)
    env = gym.make('Reacher-v4')
    
    # Test random policy
    print("Testing random policy...")
    random_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        episode_reward = 0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(episode_reward)
    
    # Test trained policy
    print("Testing trained policy...")
    trained_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        episode_reward = 0
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        trained_rewards.append(episode_reward)
    
    env.close()
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([random_rewards, trained_rewards], 
                labels=['Random Policy', 'Trained Policy'])
    plt.ylabel('Episode Reward')
    plt.title('Random vs Trained Policy Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Random policy mean reward: {np.mean(random_rewards):.2f}")
    print(f"Trained policy mean reward: {np.mean(trained_rewards):.2f}")
    print(f"Improvement: {np.mean(trained_rewards) - np.mean(random_rewards):.2f}")

if __name__ == "__main__":
    # Plot training curves
    plot_training_curves()
    
    # Evaluate trained model with visualization
    evaluate_trained_model(render=True)
    
    # Compare policies
    compare_random_vs_trained()
