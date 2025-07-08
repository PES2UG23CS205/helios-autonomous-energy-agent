# src/data_ingestion/api_clients.py
import requests
import numpy as np

# !! IMPORTANT: Replace with your actual API key !!
OPENWEATHER_API_KEY = "Your_API_KEY"

def get_weather_forecast(lat=34.05, lon=-118.24):
    """Fetches a 24-hour temperature forecast from OpenWeatherMap."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Get hourly forecast for the next 24 hours
        return [item['main']['temp'] for item in data['list'][:24]]
    except Exception as e:
        print(f"Error fetching weather data: {e}. Using default values.")
        return [15 + 5 * np.sin(i * np.pi / 12) for i in range(24)] # Default sinusoidal data

def get_simulated_energy_prices():
    """Simulates a dynamic energy price forecast with morning and evening peaks."""
    base_price = 0.10  # $/kWh
    peak_price = 0.35  # $/kWh
    prices = [base_price] * 24
    # Morning peak (7-10 AM)
    for i in range(7, 11): prices[i] = peak_price - (0.05 * (i-7))
    # Evening peak (5-9 PM)
    for i in range(17, 22): prices[i] = peak_price - (0.04 * (i-17))
    return prices
