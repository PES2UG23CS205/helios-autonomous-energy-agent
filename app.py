import streamlit as st
import pandas as pd
import torch
import time
import traceback

# Import all our custom modules
from src.environment.building_env import BuildingEnv
from src.agent.dqn_agent import DQNAgent # DQNAgent and QNetwork were in the same file
from src.agent.rag_system import RAGSystem
from src.data_ingestion.api_clients import get_weather_forecast, get_simulated_energy_prices

# --- App Configuration ---
st.set_page_config(
    page_title="Helios Energy Agent",
    page_icon="⚡",
    layout="wide"
)

# --- Model Loading (Cached for efficiency) ---
@st.cache_resource(show_spinner="Loading AI models and environment...")
def load_all_systems():
    """Loads the RL Agent, the RAG system, and the simulation environment."""
    
    # --- Load RAG System First (The Knowledge Brain) ---
    rag_system = RAGSystem()
    
    # --- Load RL Agent (The Simulation Brain) ---
    try:
        # We need some dummy values just to initialize the agent class structure
        dummy_state_size = 4 
        dummy_action_size = 3
        agent = DQNAgent(dummy_state_size, dummy_action_size)
        # Load the actual trained weights
        agent.policy_net.load_state_dict(torch.load("models/dqn_agent.pth"))
        agent.epsilon = 0.0 # Set to evaluation mode (no random actions)
    except FileNotFoundError:
        st.error("Could not find the trained model file 'models/dqn_agent.pth'. Please run `train.py` first.", icon="🚨")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the RL agent: {e}", icon="🚨")
        traceback.print_exc()
        st.stop()
        
    return agent, rag_system

# Load the core AI models once when the app starts
agent, rag_system = load_all_systems()

st.title("⚡ Helios - Autonomous Energy Management Agent")

# --- Create Tabs for Different Functionalities ---
tab1, tab2 = st.tabs(["▶️ Simulation Dashboard", "💬 Chat with Agent"])


# ==============================================================================
# --- TAB 1: SIMULATION DASHBOARD ---
# ==============================================================================
with tab1:
    st.header("Live 24-Hour Simulation")
    
    col_sim, col_config = st.columns([3, 1])

    with col_config:
        st.subheader("Simulation Controls")
        
        comfort_min = st.slider("Min Comfort Temp (°C)", 18, 22, 20, key="sim_comfort_min")
        comfort_max = st.slider("Max Comfort Temp (°C)", 23, 28, 24, key="sim_comfort_max")
        initial_temp = st.slider("Initial Building Temp (°C)", 15, 30, 22, key="sim_initial_temp")
        
        # Button to refresh data for more dynamic simulations
        if st.button("🔄 Get New Forecasts"):
            st.session_state.weather_data = get_weather_forecast()
            st.session_state.price_data = get_simulated_energy_prices()
            st.toast("Fetched new random forecast data!")

        run_button = st.button("Run 24-Hour Simulation")

    # Initialize forecasts in session_state if they don't exist
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = get_weather_forecast()
    if 'price_data' not in st.session_state:
        st.session_state.price_data = get_simulated_energy_prices()

    with col_sim:
        if run_button:
            # Create a new environment with the current slider settings
            st.info(f"Starting new simulation: Comfort [{comfort_min}°C-{comfort_max}°C], Start Temp: {initial_temp}°C")
            env = BuildingEnv(
                comfort_range=(comfort_min, comfort_max),
                initial_temp=initial_temp,
                outside_temp_forecast=st.session_state.weather_data,
                energy_price_forecast=st.session_state.price_data
            )
            
            with st.spinner("Asking RAG agent for a strategic daily briefing..."):
                query = "Based on all available documentation, what is our integrated strategy for handling a day with high price peaks in the evening?"
                initial_insight = rag_system.query(query)
                st.info(f"**🤖 Agent's Daily Briefing:** {initial_insight}", icon="🧠")
            
            st.markdown("---")

            # Placeholders for live metrics and charts
            metrics_cols = st.columns(4)
            current_hour_ph, current_temp_ph, current_action_ph, total_cost_ph = [c.empty() for c in metrics_cols]
            temp_chart_ph, cost_chart_ph = st.empty(), st.empty()

            # Run the simulation loop with the new environment
            state, _ = env.reset()
            history = []
            
            for hour in range(24):
                action_idx = agent.act(state)
                action_map = {0: "Off", 1: "Heat", 2: "Cool"}
                
                next_state, _, _, _, info = env.step(action_idx)
                state = next_state

                # Update metrics and charts
                current_hour_ph.metric("Hour", f"{hour}:00")
                current_temp_ph.metric("Building Temp", f"{env.current_temp:.1f} °C")
                current_action_ph.metric("HVAC Action", action_map[action_idx])
                total_cost_ph.metric("Total Cost", f"${env.total_cost:.2f}")

                history.append({
                    "Hour": hour, "Building Temp": env.current_temp, "Comfort Min": comfort_min, 
                    "Comfort Max": comfort_max, "Outside Temp": st.session_state.weather_data[hour],
                    "Energy Price": st.session_state.price_data[hour], "Hourly Cost": info['cost']
                })
                df = pd.DataFrame(history)

                temp_chart_ph.line_chart(df.set_index('Hour')[["Building Temp", "Outside Temp", "Comfort Min", "Comfort Max"]])
                cost_chart_ph.bar_chart(df.set_index('Hour')[["Energy Price", "Hourly Cost"]])
                
                time.sleep(0.2)
            
            st.success("Simulation Complete!")

# ==============================================================================
# --- TAB 2: CHAT INTERFACE ---
# =_============================================================================
with tab2:
    st.header("Chat with the Helios Knowledge Base")
    st.markdown("Ask about HVAC procedures, grid rules, and energy saving strategies.")

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display prior chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get new user input with an updated example prompt
    prompt = st.chat_input("e.g., How does HVAC maintenance impact grid compliance?")

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_system.query(prompt)
                st.markdown(response)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})