"""
Simple D4RL environment registration for gymnasium compatibility
"""

try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np

class D4RLWrapper(gym.Env):
    """Simple wrapper to make D4RL environment IDs work with gymnasium"""
    
    def __init__(self, env_name):
        # For our offline RL case, we just need the env spec without actual simulation
        self.env_name = env_name
        
        # Define environment specs for the ones we'll use
        if "hopper" in env_name:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        elif "halfcheetah" in env_name:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        elif "walker2d" in env_name:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        elif "ant" in env_name:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
    
    def reset(self, **kwargs):
        """Return a dummy observation"""
        obs = self.observation_space.sample()
        return obs, {}
    
    def step(self, action):
        """Return dummy step data"""
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

# Register D4RL environments
d4rl_envs = [
    'hopper-medium-v2',
    'hopper-medium-expert-v2', 
    'hopper-expert-v2',
    'hopper-random-v2',
    'halfcheetah-medium-v2',
    'halfcheetah-medium-expert-v2',
    'halfcheetah-expert-v2', 
    'halfcheetah-random-v2',
    'walker2d-medium-v2',
    'walker2d-medium-expert-v2',
    'walker2d-expert-v2',
    'walker2d-random-v2',
    'ant-medium-v2',
    'ant-medium-expert-v2',
    'ant-expert-v2',
    'ant-random-v2'
]

for env_id in d4rl_envs:
    try:
        gym.register(
            id=env_id,
            entry_point='d4rl_envs:D4RLWrapper',
            kwargs={'env_name': env_id}
        )
    except gym.error.Error:
        # Already registered
        pass 