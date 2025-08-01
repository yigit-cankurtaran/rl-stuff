#!/usr/bin/env python3
"""Test MuJoCo + Gymnasium setup with uv"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import mujoco
        print(f"✅ MuJoCo version: {mujoco.__version__}")
    except ImportError as e:
        print(f"❌ MuJoCo import failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        print(f"✅ Gymnasium version: {gym.__version__}")
    except ImportError as e:
        print(f"❌ Gymnasium import failed: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO
        import stable_baselines3
        print(f"✅ Stable-Baselines3 version: {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"❌ Stable-Baselines3 import failed: {e}")
        return False
    
    return True

def test_mujoco_env():
    """Test MuJoCo environment creation"""
    print("\nTesting MuJoCo environment...")
    
    try:
        import gymnasium as gym
        
        # List available MuJoCo environments
        mujoco_envs = [env for env in gym.envs.registry.keys() if 'mujoco' in str(gym.envs.registry[env].entry_point).lower()]
        print(f"Found {len(mujoco_envs)} MuJoCo environments")
        
        # Test creating Reacher
        env = gym.make('Reacher-v4')
        print(f"✅ Created {env.spec.id}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test basic functionality
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Sample reward: {reward:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_training_setup():
    """Test basic training setup"""
    print("\nTesting training setup...")
    
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        
        # Create environment
        env = gym.make('Reacher-v4')
        
        # Create model (don't train, just test creation)
        model = PPO('MlpPolicy', env, verbose=0)
        print("✅ PPO model created successfully")
        
        # Test prediction
        obs, info = env.reset()
        action, _states = model.predict(obs)
        print(f"✅ Model prediction works, action shape: {action.shape}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Training setup test failed: {e}")
        return False

def show_system_info():
    """Show system information"""
    print(f"\nSystem Info:")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        import mujoco
        # Check if we can access MuJoCo model info
        print(f"MuJoCo binary path: {mujoco.mj_versionString()}")
    except:
        pass

if __name__ == "__main__":
    print("=== MuJoCo + Gymnasium + SB3 Setup Test ===")
    
    show_system_info()
    
    success = True
    success &= test_imports()
    success &= test_mujoco_env()
    success &= test_training_setup()
    
    if success:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nYou can now use the training scripts from before:")
        print("- python train_agent.py")
        print("- python visualize_results.py")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
        print("\nTroubleshooting:")
        print("- Make sure you activated the virtual environment")
        print("- Try: uv add --upgrade gymnasium[mujoco]")
        print("- Check that your system MuJoCo installation is working")
