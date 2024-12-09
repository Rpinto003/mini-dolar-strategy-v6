# src/models/trading_env.py

import gym
import numpy as np
from gym import spaces

class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    def __init__(self):
        super().__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Market features space
        num_features = 10
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_features,), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.position = 0
        self.current_observation = np.zeros(self.observation_space.shape[0])
        return self.current_observation
        
    def step(self, action):
        reward = 0
        done = False
        info = {}
        
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            reward = 1
                
        self.current_step += 1
        done = self.current_step >= 1000
            
        return self.current_observation, reward, done, info
        
    def render(self, mode='human'):
        pass

    def _get_observation(self):
        """Helper method to get current market observation"""
        # Aqui você implementaria a lógica para criar as features do mercado
        return np.zeros(self.observation_space.shape[0])  # Placeholder