import gymnasium as gym
import numpy as np

def test_mujoco_installation():
    """Test MuJoCo installation and rendering"""
    print("Testing MuJoCo installation...")
    
    # Create environment
    env = gym.make('Reacher-v4', render_mode='human')
    
    print(f"Environment: {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few random steps
    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("✅ MuJoCo installation successful!")

if __name__ == "__main__":
    test_mujoco_installation()
