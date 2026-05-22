import requests
import json
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "rainfall_cache.json")

def _build_pond_coords():
    coords = {}
    try:
        _df = pd.read_csv(os.path.join(DATA_DIR, "pond_data.csv"))
        for _, r in _df.iterrows():
            coords[r["pond_id"]] = (float(r["latitude"]), float(r["longitude"]))
    except Exception:
        pass
    return coords

def run():
    # Defensive programming: Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "pond_data.csv"))
    except FileNotFoundError:
        print("❌ Error: pond_data.csv not found. Ensure Phase 1 ran first.")
        return

    try:
        with open(CACHE_PATH, 'r') as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    POND_COORDS = _build_pond_coords()
    # Extract coordinates in exact sequential order for multi-location mapping
    pids = list(POND_COORDS.keys())
    lats = [str(POND_COORDS[pid][0]) for pid in pids]
    lons = [str(POND_COORDS[pid][1]) for pid in pids]

    # Combine all coordinates into a single batched Open-Meteo URL request
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={','.join(lats)}&longitude={','.join(lons)}"
        f"&daily=precipitation_sum&forecast_days=3&timezone=Asia/Kolkata"
    )

    rainfall_source = "live"
    try:
        print("📡 Querying batched 72-hour weather forecast arrays...")
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        response_data = r.json()
        
        # Open-Meteo returns a list of dictionaries if multiple coordinates are sent
        if not isinstance(response_data, list):
            response_data = [response_data]

        for idx, pid in enumerate(pids):
            location_data = response_data[idx]
            raw_precip = location_data['daily']['precipitation_sum']
            
            # Safeguard against unexpected Null/None types in the API payload
            mm = sum(x if x is not None else 0.0 for x in raw_precip)
            mm = round(mm, 1)
            
            print(f"  {pid}: {mm}mm from Live API Sync")
            df.loc[df['pond_id'] == pid, 'rainfall_forecast_mm'] = mm
            cache[pid] = mm

    except Exception as e:
        rainfall_source = "cache"
        print(f"⚠️ Live API payload failed ({e}). Reverting to fallbacks...")
        for pid in pids:
            mm = cache.get(pid, 50.0)  # Standard baseline fallback metric
            print(f"  {pid}: {mm}mm extracted from localized storage cache")
            df.loc[df['pond_id'] == pid, 'rainfall_forecast_mm'] = mm
            cache[pid] = mm
    df['rainfall_source'] = rainfall_source

    # Commit state changes cleanly back to CSV data engine layer
    df.to_csv(os.path.join(DATA_DIR, "pond_data.csv"), index=False)
    
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, 'r') as f:
        content = f.read()
    new_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M IST")
    import re
    content = re.sub(r'LAST_FETCH = ".*?"', f'LAST_FETCH = "{new_ts}"', content)
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("✅ Hydrological forecast arrays updated in target fields.")

if __name__ == "__main__":
    run()