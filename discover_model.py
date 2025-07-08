import os
import requests
from dotenv import load_dotenv

# --- SETUP: Make sure to have a .env file with your token ---
# In your project folder, create a file named `.env`
# and put this line in it:
# HUGGINGFACEHUB_API_TOKEN="hf_YOUR_SECRET_TOKEN"
load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not API_TOKEN:
    print("CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
    print("Please create a .env file in this directory with your token.")
    exit()

# --- THE DISCOVERY FUNCTION ---
def get_available_models():
    """Fetches a list of recommended models from the Hugging Face Hub."""
    print("Querying Hugging Face Hub for available models...")
    api_url = "https://api-inference.huggingface.co/models"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status() # This will raise an error for 4xx/5xx responses
        models = response.json()
        
        print("\n--- Found Available Models ---")
        
        # We will filter for good candidates for our task
        text_generation_candidates = []
        
        for model in models:
            # Look for models suitable for text generation or instruction following
            if model.get("pipeline_tag") in ["text-generation", "text2text-generation"] and "instruct" in model.get("modelId", "").lower():
                text_generation_candidates.append(model)
        
        if not text_generation_candidates:
            print("Could not automatically find good instruction-following models. Showing first 10 generic models instead.")
            for model in models[:10]:
                 print(f"- Model ID: {model.get('modelId')}, Task: {model.get('pipeline_tag')}")
        else:
            print("--- Top Recommended Models for Text Generation ---")
            for model in text_generation_candidates[:10]: # Show top 10 candidates
                print(f"- Model ID: {model.get('modelId')}, Task: {model.get('pipeline_tag')}")

    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP ERROR: Failed to query the Hugging Face API.")
        print(f"Status Code: {e.response.status_code}")
        print(f"Details: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the discovery
get_available_models()