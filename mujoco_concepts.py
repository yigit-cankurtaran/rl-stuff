import gymnasium as gym
import numpy as np

def explore_mujoco_concepts():
    """Explore MuJoCo-specific concepts in Reacher-v4"""
    
    env = gym.make('Reacher-v4')
    obs, info = env.reset()
    
    print("=== MUJOCO ENVIRONMENT CONCEPTS ===\n")
    
    # 1. OBSERVATION AND ACTION SPACES
    print("1. OBSERVATION AND ACTION SPACES:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Action shape: {env.action_space.shape}")
    print(f"   Action bounds: [{env.action_space.low}, {env.action_space.high}]")
    
    # 2. OBSERVATION BREAKDOWN
    print(f"\n2. OBSERVATION BREAKDOWN (Reacher-v4 has {len(obs)} observations):")
    print("   Reacher observations typically include:")
    print("   - Joint angles (cos, sin of each joint)")
    print("   - Joint velocities") 
    print("   - Target position relative to fingertip")
    print(f"   Current observation: {obs}")
    
    # 3. MUJOCO MODEL COMPONENTS
    print("\n3. MUJOCO MODEL COMPONENTS:")
    
    # Access the MuJoCo model
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    print(f"   Bodies: {model.nbody} bodies")
    for i in range(model.nbody):
        body_name = model.body(i).name or f"body_{i}"
        print(f"     - {body_name}")
    
    print(f"   Joints: {model.njnt} joints")
    for i in range(model.njnt):
        joint_name = model.joint(i).name or f"joint_{i}"
        joint_type = model.joint(i).type
        print(f"     - {joint_name} (type: {joint_type})")
    
    print(f"   Actuators: {model.nu} actuators")
    for i in range(model.nu):
        actuator_name = model.actuator(i).name or f"actuator_{i}"
        print(f"     - {actuator_name}")
    
    # 4. CURRENT STATE INFORMATION
    print(f"\n4. CURRENT STATE INFORMATION:")
    print(f"   Joint positions (qpos): {data.qpos}")
    print(f"   Joint velocities (qvel): {data.qvel}")
    print(f"   Control inputs (ctrl): {data.ctrl}")
    
    # 5. DEMONSTRATE PHYSICS SIMULATION
    print(f"\n5. PHYSICS SIMULATION DEMO:")
    print("   Applying random actions and observing state changes...")
    
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {step + 1}:")
        print(f"     Action: {action}")
        print(f"     Reward: {reward:.4f}")
        print(f"     New joint positions: {data.qpos}")
        print(f"     New joint velocities: {data.qvel}")
        
        if terminated or truncated:
            break
    
    env.close()

def explain_reacher_specifics():
    """Explain Reacher-v4 specific details"""
    print("\n=== REACHER-V4 SPECIFICS ===\n")
    
    print("TASK DESCRIPTION:")
    print("- 2-link arm reaching to a target")
    print("- Goal: Move fingertip to red target sphere")
    print("- Reward: Negative distance to target + control penalty")
    
    print("\nCOMPONENTS:")
    print("- Bodies: ground, link0, link1, fingertip, target")
    print("- Joints: 2 revolute joints (shoulder, elbow)")
    print("- Actuators: 2 torque motors for each joint")
    
    print("\nOBSERVATION SPACE (11-dim vector):")
    print("- cos(joint angles): 2 values")
    print("- sin(joint angles): 2 values") 
    print("- joint velocities: 2 values")
    print("- target position (x,y): 2 values")
    print("- fingertip to target vector: 3 values")
    
    print("\nACTION SPACE:")
    print("- 2D continuous actions [-1, 1]")
    print("- Each action controls torque for one joint")
    
    print("\nREWARD FUNCTION:")
    print("- reward = -distance_to_target - control_cost")
    print("- distance_to_target: L2 distance between fingertip and target")
    print("- control_cost: small penalty for large actions (encourages efficiency)")

def mujoco_best_practices():
    """MuJoCo best practices for RL"""
    print("\n=== MUJOCO RL BEST PRACTICES ===\n")
    
    print("1. ENVIRONMENT SETUP:")
    print("   - Use frame_skip to reduce control frequency")
    print("   - Consider action/observation normalization")
    print("   - Set appropriate episode length limits")
    
    print("\n2. HYPERPARAMETER TUNING:")
    print("   - Start with default SB3 hyperparameters")
    print("   - Adjust learning rate based on environment complexity")
    print("   - Use higher batch sizes for continuous control")
    
    print("\n3. COMMON ISSUES:")
    print("   - Exploding gradients: reduce learning rate")
    print("   - Poor exploration: adjust entropy coefficient")
    print("   - Unstable training: increase batch size or use frame skip")
    
    print("\n4. ALGORITHM CHOICES:")
    print("   - PPO: Good general-purpose algorithm, stable")
    print("   - SAC: Better sample efficiency, handles sparse rewards")
    print("   - TD3: Alternative to SAC, more deterministic")

if __name__ == "__main__":
    explore_mujoco_concepts()
    explain_reacher_specifics()
    mujoco_best_practices()
