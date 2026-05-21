import os

# API Keys 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_PASTE_YOUR_KEY_HERE")
GROQ_MODEL = "llama3-8b-8192"

# App
APP_TITLE = "CoalWatch — Ash Pond Breach Early Warning System"

# Risk Thresholds
CRITICAL_THRESHOLD = 0.70
ELEVATED_THRESHOLD = 0.40

# Map
MAP_CENTER_LAT = 23.2
MAP_CENTER_LON = 82.64
MAP_ZOOM = 7

#  Disclaimers 
RETRO_DISCLAIMER = (
    "Simulated retrospective analysis based on Singrauli 2020 monsoon onset "
    "records and Sasan UMPP site parameters. Not derived from live satellite imagery."
)
LAST_FETCH = "2024-07-15 09:00 IST"

# Feature Normalisation Maxima 
NDWI_MAX = 0.55
NDVI_STRESS_MAX = 1.0
SLOPE_MAX = 25.0
RAINFALL_MAX = 180.0
SAR_MAX = 0.30

# DII Weights (must sum to 1.0) 
W_NDWI = 0.28
W_NDVI = 0.24
W_SLOPE = 0.19
W_RAINFALL = 0.15
W_PROXIMITY = 0.09
W_SAR = 0.05

assert round(W_NDWI + W_NDVI + W_SLOPE + W_RAINFALL + W_PROXIMITY + W_SAR, 10) == 1.0, \
    "DII weights must sum to 1.0"

# Paths 
MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "model")

# ML Features 
FEATURE_COLS = [
    'ndwi_score', 'ndvi_score', 'slope_degrees',
    'rainfall_forecast_mm', 'breach_proximity_score'
]