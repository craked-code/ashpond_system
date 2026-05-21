import planetary_computer
import pystac_client
import stackstac
import numpy as np
import pandas as pd
import os

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
            bounds_latlon=(lon-0.005, lat-0.005, lon+0.005, lat+0.005),
            resolution=10,
            epsg=32644,
        )

        green = stack.sel(band="B03").values[0].astype(float)
        red   = stack.sel(band="B04").values[0].astype(float)
        nir   = stack.sel(band="B08").values[0].astype(float)

        if green.ndim == 0:
            return None, None

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
            bounds_latlon=(lon-0.05, lat-0.05, lon+0.05, lat+0.05),
            resolution=10,
            epsg=32644,
        )

        vv = stack.values[0].astype(float)
        backscatter = float(np.nanmean(vv))
        return round(backscatter, 4)

    except Exception as e:
        print(f"    fetch_sentinel1_sar ERROR: {type(e).__name__}: {e}")
        return None
    

def run():
    catalog = get_catalog()
    df = pd.read_csv("data/pond_data.csv")

    for _, row in df.iterrows():
        pid = row["pond_id"]
        lat = row["latitude"]
        lon = row["longitude"]
        print(f"\nFetching {pid} ({row['pond_name']})...")

        ndwi, ndvi = fetch_sentinel2(lat, lon, catalog)
        if ndwi is None or np.isnan(ndwi):
            print(f"  Sentinel-2 FAILED for {pid}, keeping existing")
        else:
            df.loc[df['pond_id']==pid,'ndwi_score'] = ndwi
            df.loc[df['pond_id']==pid,'ndvi_score'] = ndvi
            print(f"  S2: NDWI={ndwi}, NDVI={ndvi}")

        sar = fetch_sentinel1_sar(lat, lon, catalog)
        if sar is None or np.isnan(sar):
            print(f"  Sentinel-1 FAILED for {pid}, keeping existing")
        else:
            df.loc[df['pond_id']==pid,'sar_backscatter'] = sar
            print(f"  S1 SAR: backscatter={sar}")

    df.to_csv("data/pond_data.csv", index=False)
    print("\nDone. pond_data.csv updated with real satellite values.")

if __name__ == "__main__":
    run()