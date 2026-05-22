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

def load_inspection_log():
    log_path = os.path.join(DATA_DIR, "inspection_log.csv")
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=[
            'timestamp', 'pond_id', 'pond_name',
            'inspector_name', 'notes', 'photo_filename'
        ])
    return pd.read_csv(log_path)


def save_inspection(pond_id, pond_name, inspector_name, notes, photo_filename=""):
    log_path = os.path.join(DATA_DIR, "inspection_log.csv")
    photos_dir = os.path.join(DATA_DIR, "photos")
    os.makedirs(photos_dir, exist_ok=True)

    new_row = pd.DataFrame([{
        'timestamp':      pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        'pond_id':        pond_id,
        'pond_name':      pond_name,
        'inspector_name': inspector_name,
        'notes':          notes,
        'photo_filename': photo_filename,
    }])

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        pd.concat([existing, new_row], ignore_index=True).to_csv(log_path, index=False)
    else:
        new_row.to_csv(log_path, index=False)