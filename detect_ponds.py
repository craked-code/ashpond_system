import numpy as np
import pandas as pd
import os
from scipy import ndimage
import planetary_computer
import pystac_client
import stackstac

PLANT_COORDS = [
    {"plant": "Sasan UMPP",          "region": "Singrauli", "lat": 23.966, "lon": 82.625},
    {"plant": "Vindhyachal STPS",    "region": "Singrauli", "lat": 24.097, "lon": 82.674},
    {"plant": "Singrauli Super TPS", "region": "Singrauli", "lat": 24.105, "lon": 82.706},
    {"plant": "Rihand STPS",         "region": "Singrauli", "lat": 24.027, "lon": 82.790},
    {"plant": "NTPC Korba",          "region": "Korba",     "lat": 22.386, "lon": 82.682},
    {"plant": "CSEB Korba East",     "region": "Korba",     "lat": 22.371, "lon": 82.738},
    {"plant": "CSEB Korba West",     "region": "Korba",     "lat": 22.360, "lon": 82.700},
    {"plant": "NTPC Sipat",          "region": "Bilaspur",  "lat": 22.132, "lon": 82.292},
]

NDWI_THRESHOLD  = 0.10
MIN_POND_PIXELS = 100    # ~5 hectares at 20m resolution
MAX_POND_PIXELS = 50000  # ~500 hectares
SEARCH_RADIUS   = 0.08   # ~9km

def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

def fetch_ndwi_grid(lat, lon, catalog):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime="2024-04-01/2024-07-15",
        query={"eo:cloud_cover": {"lt": 15}},
        max_items=3,
    )
    items = list(search.items())
    if not items:
        return None, None, None

    stack = stackstac.stack(
        items[:1],
        assets=["B03", "B08"],
        bounds_latlon=(lon-SEARCH_RADIUS, lat-SEARCH_RADIUS,
                       lon+SEARCH_RADIUS, lat+SEARCH_RADIUS),
        resolution=20,
        epsg=32644,
    )

    green = stack.sel(band="B03").values[0].astype(float)
    nir   = stack.sel(band="B08").values[0].astype(float)
    ndwi  = (green - nir) / (green + nir + 1e-10)

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    lons_utm = stack.x.values
    lats_utm = stack.y.values

    return ndwi, lons_utm, lats_utm, transformer

def detect_ponds_near_plant(plant, catalog):
    lat, lon = plant["lat"], plant["lon"]
    ndwi, lons, lats, transformer = fetch_ndwi_grid(lat, lon, catalog)

    if ndwi is None:
        print(f"  No satellite data found for {plant['plant']}")
        return []

    water_mask = ndwi > NDWI_THRESHOLD
    if water_mask.sum() < MIN_POND_PIXELS:
        water_mask = ndwi > 0.05
    labeled, n_features = ndimage.label(water_mask)

    ponds = []
    for i in range(1, n_features + 1):
        component = labeled == i
        size = int(component.sum())
        if MIN_POND_PIXELS <= size <= MAX_POND_PIXELS:
            cy, cx = ndimage.center_of_mass(component)
            utm_x = float(np.interp(cx, range(len(lons)), lons))
            utm_y = float(np.interp(cy, range(len(lats)), lats))
            pond_lon, pond_lat = transformer.transform(utm_x, utm_y)
            area_ha  = round(size * 20 * 20 / 10000, 1)
            ponds.append({
                "plant":   plant["plant"],
                "region":  plant["region"],
                "lat":     round(pond_lat, 5),
                "lon":     round(pond_lon, 5),
                "area_ha": area_ha,
            })

    ponds.sort(key=lambda p: p["area_ha"], reverse=True)
    return ponds[:4]

def run():
    catalog = get_catalog()
    all_ponds = []

    for plant in PLANT_COORDS:
        print(f"\nScanning {plant['plant']}...")
        try:
            ponds = detect_ponds_near_plant(plant, catalog)
            print(f"  Found {len(ponds)} candidate pond(s)")
            for p in ponds:
                print(f"    {p['lat']:.4f}, {p['lon']:.4f} — {p['area_ha']:.0f} ha")
            all_ponds.extend(ponds)
        except Exception as e:
            print(f"  FAILED: {e}")

    if not all_ponds:
        print("\nNo ponds detected. Check satellite connectivity.")
        return
    
    unique_ponds = []
    for pond in all_ponds:
        is_duplicate = False
        for existing in unique_ponds:
            if abs(pond["lat"] - existing["lat"]) < 0.01 and \
               abs(pond["lon"] - existing["lon"]) < 0.01:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_ponds.append(pond)
    all_ponds = unique_ponds
    print(f"\nAfter deduplication: {len(all_ponds)} unique ponds")

    rows = []
    for i, pond in enumerate(all_ponds[:20], 1):
        pid    = f"P{i:03d}"
        letter = chr(64 + (i % 26 or 26))
        rows.append({
            "pond_id":                pid,
            "pond_name":              f"{pond['plant']} Pond {letter}",
            "plant_name":             pond["plant"],
            "region":                 pond["region"],
            "latitude":               pond["lat"],
            "longitude":              pond["lon"],
            "ndwi_score":             0.0,
            "ndvi_score":             0.5,
            "slope_degrees":          10,
            "rainfall_forecast_mm":   50,
            "breach_proximity_score": 0.5,
            "sar_backscatter":        0.05,
            "dii_score":              0.0,
            "risk_category":          "STABLE",
            "top_risk_factor":        "Awaiting sensor data",
            "last_updated":           "pending",
            "dii_score_prev":         0.0,
            "dii_trend":              "→",
            "breach_probability":     0.0,
            "dii_lower":              0.0,
            "dii_upper":              0.0,
            "mahal_distance":         0.0,
            "profile_anomaly":        False,
            "dii_mean":               0.0,
        })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/pond_data.csv", index=False)
    print(f"\nWrote {len(df)} ponds to data/pond_data.csv")
    print("Next: run fetch_satellite_data.py → fetch_rainfall.py → compute_dii.py → ml_models.py")

if __name__ == "__main__":
    run()