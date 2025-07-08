# train.py
import torch
from src.environment.building_env import BuildingEnv
from src.agent.dqn_agent import DQNAgent
from src.data_ingestion.api_clients import get_weather_forecast, get_simulated_energy_prices

# --- Hyperparameters ---
EPISODES = 500
BATCH_SIZE = 32
MODEL_SAVE_PATH = "models/dqn_agent.pth"

# --- Setup ---
weather_data = get_weather_forecast()
price_data = get_simulated_energy_prices()

env = BuildingEnv(outside_temp_forecast=weather_data, energy_price_forecast=price_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# --- Training Loop ---
for e in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward

        if done:
            agent.update_target_net()
            print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
            
# --- Save Model ---
torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")