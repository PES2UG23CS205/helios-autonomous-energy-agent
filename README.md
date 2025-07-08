# ⚡ Helios - Autonomous Energy Management Agent

 
*A snapshot of the Helios Simulation Dashboard in action.*
![Screenshot 2025-07-08 150754](https://github.com/user-attachments/assets/f2f7a463-ec4c-4e6d-8551-6af58d10bd5a)


## 📖 Introduction

**Helios** is an advanced AI-powered system designed to intelligently manage energy consumption in buildings. It acts as a digital "brain" that moves beyond simple timers and thermostats, creating a dynamic, responsive, and cost-effective energy strategy.

This project tackles the common problem of energy inefficiency in buildings, which often suffer from outdated management systems that cannot adapt to real-time variables. By leveraging two distinct AI agents, Helios optimizes for cost, comfort, and sustainability.

---

## 🎯 Problem Statement

Modern buildings and energy grids face significant challenges:
- **Inefficient Energy Usage:** Outdated systems lead to excessive consumption and high operational costs.
- **Lack of Real-time Adaptability:** Traditional systems cannot react to dynamic factors like changing weather or fluctuating energy prices.
- **Missed Sustainability Opportunities:** Inability to participate in grid-level programs like demand response leads to increased carbon emissions and financial penalties.

The goal of this project is to develop an autonomous agent that can monitor, analyze, and dynamically adjust a building's energy usage to address these challenges head-on.

---

## 💡 Solution Approach

Helios employs a powerful, two-agent AI architecture to create a comprehensive solution:

**1. The Reinforcement Learning (RL) Agent - The "Pilot"**
   - **What it does:** This agent is the real-time operator. It's responsible for making second-by-second decisions on how to control the building's HVAC (Heating, Ventilation, and Air Conditioning) system.
   - **How it works:** Using a **Deep Q-Network (DQN)**, the agent was trained over hundreds of simulated days. It learned through trial and error, receiving digital "rewards" for saving energy and maintaining comfort, and "penalties" for wasting resources.
   - **Its Goal:** To learn the optimal strategy that perfectly balances three competing objectives: minimizing energy costs, maximizing occupant comfort, and ensuring grid stability. You can see its intelligent decision-making in the Simulation Dashboard, where it pre-cools the building before price spikes.

**2. The Retrieval-Augmented Generation (RAG) Agent - The "Advisor"**
   - **What it does:** This agent acts as the on-demand knowledge expert. It understands the building's technical manuals, grid regulations, and operational best practices.
   - **How it works:** It uses a state-of-the-art **RAG** pipeline. When asked a question, it first *retrieves* the most relevant information from its knowledge base (`.txt` files) and then uses a locally-hosted Large Language Model (**Ollama with Llama 3**) to *generate* a comprehensive, human-like answer based *only* on those facts.
   - **Its Goal:** To provide strategic advice and ensure all automated actions comply with documented rules. It provides the "Daily Briefing" in the simulation and allows users to ask complex questions in plain English via the "Chat with Agent" tab.

Together, these two agents create a system where a smart "Pilot" executes the best actions, while an expert "Advisor" provides the strategic oversight.

---

## 🛠️ Tech Stack

This project utilizes a modern, end-to-end Python-based AI and data science stack.

- **Programming Language:** `Python 3.9+`
- **AI & Machine Learning:**
    - `PyTorch`: For building the neural network brain of the RL agent.
    - `Gymnasium (formerly OpenAI Gym)`: For creating the custom simulated building environment.
    - `LangChain`: To orchestrate the complex RAG pipeline.
    - `Ollama`: For running the powerful Llama 3 Large Language Model entirely locally and for free.
    - `FAISS (Facebook AI Similarity Search)`: For ultra-fast vector search in the RAG knowledge base.
    - `Sentence-Transformers`: For converting text documents into numerical vectors (embeddings).
- **Data Handling & APIs:** `Pandas`, `NumPy`, `Requests`.
- **Dashboard & UI:** `Streamlit`.
- **Version Control:** `Git` & `GitHub`.

---

## 🚀 How to Run Locally

Follow these steps to set up and run the Helios agent on your own machine.

### Prerequisites

- Python 3.9+
- Git
- [Ollama](https://ollama.com/) installed on your system.

### 1. Set Up the Local LLM (Ollama)

First, you need to download and run the local Large Language Model. Open a standard terminal (not in your project folder yet) and run:
```bash
ollama pull llama3:8b
```
This will download the Llama 3 8B model to your machine. Ollama will run as a background service.

### 2. Clone the Repository

Clone this project to your local machine:
```bash
git clone https://github.com/PES2UG23CS205/helios-autonomous-energy-agent.git
cd helios-autonomous-energy-agent
```

### 3. Set Up the Python Virtual Environment

Create and activate a virtual environment to keep dependencies clean:
```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 4. Install Dependencies

Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 5. Train the Reinforcement Learning Agent (One-time step)

Before you can run the simulation, you need to train the RL agent. This will create the `dqn_agent.pth` model file.
```bash
python train.py
```
This process will take a few minutes as it runs through 500 simulated days of learning.

### 6. Run the Streamlit Application

You're all set! Launch the Helios dashboard with this command:
```bash
streamlit run app.py
```
The application will open in a new tab in your web browser. You can now run simulations and chat with the agent!

---

## 📞 Contact

This project was created by **Gowtham B**.

- **GitHub:** [PES2UG23CS205](https://github.com/PES2UG23CS205)
- **LinkedIn:** www.linkedin.com/in/gowtham-b-bb8960305


- **Email:** gowthammourya9@gmail.com

Feel free to reach out with any questions or feedback!
