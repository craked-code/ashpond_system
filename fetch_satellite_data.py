import planetary_computer
import pystac_client
import stackstac
import numpy as np
import pandas as pd
import os

POND_COORDINATES = [
    {"pond_id":"P001","lat":24.0721,"lon":82.5892},
    {"pond_id":"P002","lat":24.0698,"lon":82.5901},
    {"pond_id":"P003","lat":24.0833,"lon":82.6667},
    {"pond_id":"P004","lat":24.0812,"lon":82.6701},
    {"pond_id":"P005","lat":24.0347,"lon":82.8156},
    {"pond_id":"P006","lat":22.3460,"lon":82.6890},
    {"pond_id":"P007","lat":22.3421,"lon":82.6912},
    {"pond_id":"P008","lat":22.3612,"lon":82.7012},
    {"pond_id":"P009","lat":22.3398,"lon":82.7156},
    {"pond_id":"P010","lat":22.3201,"lon":82.6734},
]

def get_catalog():
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        _ = catalog.title
        return catalog
    except Exception as e:
        raise RuntimeError(
            f"[get_catalog] Failed to connect to Planetary Computer.\n"
            f"  Type: {type(e).__name__}\n"
            f"  Msg : {e}\n"
            f"  Checklist:\n"
            f"    1. Behind a proxy? Set HTTPS_PROXY env var.\n"
            f"    2. SSL failure? Export REQUESTS_CA_BUNDLE.\n"
            f"    3. Run: ping planetarycomputer.microsoft.com\n"
        ) from e
    

def fetch_sentinel2(lat, lon, catalog):
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects={"type":"Point","coordinates":[lon,lat]},
            datetime="2024-05-01/2024-07-15",
            query={"eo:cloud_cover":{"lt":20}},
            max_items=3,
        )
        items = list(search.items())
        if not items:
            print("    No Sentinel-2 scenes found (cloud cover or no data)")
            return None, None

        stack = stackstac.stack(
            items[:1],
            assets=["B03","B04","B08"],
            bounds=(lon-0.005, lat-0.005, lon+0.005, lat+0.005),
            resolution=10,
        ).squeeze()

        green = stack.sel(band="B03").values.astype(float)
        red   = stack.sel(band="B04").values.astype(float)
        nir   = stack.sel(band="B08").values.astype(float)

        ndwi = float(np.nanmean((green - nir) / (green + nir + 1e-10)))
        ndvi = float(np.nanmean((nir - red)   / (nir + red + 1e-10)))

        return round(ndwi, 3), round(ndvi, 3)

    except Exception as e:
        print(f"    fetch_sentinel2 ERROR: {type(e).__name__}: {e}")
        return None, None
    

def fetch_sentinel1_sar(lat, lon, catalog):
    try:
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            intersects={"type":"Point","coordinates":[lon,lat]},
            datetime="2024-05-01/2024-07-15",
            max_items=3,
        )
        items = list(search.items())
        if not items:
            print("    No Sentinel-1 scenes found")
            return None

        stack = stackstac.stack(
            items[:1],
            assets=["vv"],
            bounds=(lon-0.005, lat-0.005, lon+0.005, lat+0.005),
            resolution=10,
        ).squeeze()

        vv = stack.values.astype(float)
        backscatter = float(np.nanmean(vv))
        return round(backscatter, 4)

    except Exception as e:
        print(f"    fetch_sentinel1_sar ERROR: {type(e).__name__}: {e}")
        return None
    

def run():
    catalog = get_catalog()
    df = pd.read_csv("data/pond_data.csv")
    
    for pond in POND_COORDINATES:
        pid = pond["pond_id"]
        print(f"\nFetching {pid}...")
        
        ndwi, ndvi = fetch_sentinel2(pond["lat"], pond["lon"], catalog)
        if ndwi is None or np.isnan(ndwi):
            print(f"  Sentinel-2 FAILED for {pid}, keeping existing")
        else:
            df.loc[df['pond_id']==pid, 'ndwi_score'] = ndwi
            df.loc[df['pond_id']==pid, 'ndvi_score'] = ndvi
            print(f"  S2: NDWI={ndwi}, NDVI={ndvi}")

        sar = fetch_sentinel1_sar(pond["lat"], pond["lon"], catalog)
        if sar is None or np.isnan(sar):
            print(f"  Sentinel-1 FAILED for {pid}, keeping existing")
        else:
            df.loc[df['pond_id']==pid, 'sar_backscatter'] = sar
            print(f"  S1 SAR: backscatter={sar}")

    df.to_csv("data/pond_data.csv", index=False)
    print("\nDone. pond_data.csv updated with real satellite values.")

if __name__ == "__main__":
    run()