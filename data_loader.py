import pandas as pd
import json
import os
import pickle

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(DATA_DIR, "model")

def load_pond_data():
    """Load the main pond dataset with all AI features"""
    df = pd.read_csv(os.path.join(DATA_DIR, "pond_data.csv"))
    
    # Ensure boolean columns are properly typed
    bool_cols = ['profile_anomaly']
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    
    return df

def load_retrospective_data():
    """Load historical data for retrospective chart and change point detection"""
    df = pd.read_csv(os.path.join(DATA_DIR, "pond_retrospective.csv"))
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_rainfall_cache():
    """Load rainfall cache as fallback"""
    cache_path = os.path.join(DATA_DIR, "rainfall_cache.json")
    with open(cache_path, 'r') as f:
        return json.load(f)

def load_model():
    """Load trained cox proportional model + scaler"""
    model_path  = os.path.join(MODEL_DIR, "cox_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def get_pond_by_name_fuzzy(pond_name, df):
    """Fuzzy search for pond by name (used by chat assistant)"""
    if not pond_name or pd.isna(pond_name):
        return None
    
    keywords = pond_name.lower().split()
    for _, row in df.iterrows():
        name_lower = row['pond_name'].lower()
        if any(k in name_lower for k in keywords if len(k) > 3):
            return row
    return None