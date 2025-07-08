# src/environment/building_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BuildingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, comfort_range=(20, 24), initial_temp=22, outside_temp_forecast=[15]*24, energy_price_forecast=[0.1]*24):
        super(BuildingEnv, self).__init__()

        self.comfort_range = comfort_range
        self.initial_temp = initial_temp
        self.outside_temp_forecast = outside_temp_forecast
        self.energy_price_forecast = energy_price_forecast
        self.current_hour = 0
        
        # Action Space: 0: Off, 1: Heat, 2: Cool
        self.action_space = spaces.Discrete(3)

        # Observation Space: [current_temp, outside_temp, hour_of_day, energy_price]
        self.observation_space = spaces.Box(
            low=np.array([-10, -20, 0, 0]), 
            high=np.array([40, 50, 23, 1.0]), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = self.initial_temp
        self.current_hour = 0
        self.total_cost = 0
        self.total_comfort_penalty = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.current_temp,
            self.outside_temp_forecast[self.current_hour],
            self.current_hour,
            self.energy_price_forecast[self.current_hour]
        ], dtype=np.float32)

    def step(self, action):
        # 1. Update temperature based on action and outside temp
        outside_temp = self.outside_temp_forecast[self.current_hour]
        if action == 1: # Heat
            self.current_temp += 1.0  # Simplified physics
            energy_consumed = 2.0 # kWh
        elif action == 2: # Cool
            self.current_temp -= 1.0  # Simplified physics
            energy_consumed = 1.5 # kWh
        else: # Off
            # Drifts towards outside temp
            self.current_temp += (outside_temp - self.current_temp) * 0.1
            energy_consumed = 0.1 # Base load

        # 2. Calculate cost
        price = self.energy_price_forecast[self.current_hour]
        cost = energy_consumed * price
        self.total_cost += cost

        # 3. Calculate comfort penalty
        comfort_penalty = 0
        if self.current_temp < self.comfort_range[0]:
            comfort_penalty = self.comfort_range[0] - self.current_temp
        elif self.current_temp > self.comfort_range[1]:
            comfort_penalty = self.current_temp - self.comfort_range[1]
        
        self.total_comfort_penalty += comfort_penalty
            
        # 4. Calculate reward (we want to MINIMIZE cost and penalty, so reward is negative)
        # We heavily penalize discomfort
        reward = -cost - (comfort_penalty * 5)

        # 5. Advance time
        self.current_hour = (self.current_hour + 1) % 24
        terminated = self.current_hour == 23 # End of episode after one day

        info = {'cost': cost, 'comfort_penalty': comfort_penalty}

        return self._get_obs(), reward, terminated, False, info

    def render(self, mode='human'):
        if mode == 'human':
            print(
                f"Hour: {self.current_hour}, "
                f"Temp: {self.current_temp:.2f}°C, "
                f"Action: {['Off', 'Heat', 'Cool'][self.action_space.sample()]}, " # a sample action for rendering
                f"Cost: ${self.total_cost:.2f}, "
                f"Comfort Penalty: {self.total_comfort_penalty:.2f}"
            )