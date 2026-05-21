import pandas as pd
import config

def compute_dii_row(row):
    ndwi_n  = min(row['ndwi_score'] / config.NDWI_MAX, 1.0)
    ndvi_n  = min((1 - row['ndvi_score']) / config.NDVI_STRESS_MAX, 1.0)
    slope_n = min(row['slope_degrees'] / config.SLOPE_MAX, 1.0)
    rain_n  = min(row['rainfall_forecast_mm'] / config.RAINFALL_MAX, 1.0)
    sar_val = row.get('sar_backscatter', 0.05)
    sar_val = 0.05 if (sar_val is None or pd.isna(sar_val)) else sar_val
    sar_n   = min(sar_val / config.SAR_MAX, 1.0)
    prox    = row['breach_proximity_score']
    return round(
        config.W_NDWI      * ndwi_n  +
        config.W_NDVI      * ndvi_n  +
        config.W_SLOPE     * slope_n +
        config.W_RAINFALL  * rain_n  +
        config.W_PROXIMITY * prox    +
        config.W_SAR       * sar_n,
        3
    )

def run():
    df = pd.read_csv("data/pond_data.csv")
    df['dii_score_prev'] = df['dii_score']
    df['dii_score'] = df.apply(compute_dii_row, axis=1)

    def categorise(d):
        if d > config.CRITICAL_THRESHOLD:   return "CRITICAL"
        elif d > config.ELEVATED_THRESHOLD: return "ELEVATED"
        return "STABLE"

    df['risk_category'] = df['dii_score'].apply(categorise)
    df['dii_trend'] = df.apply(
        lambda r: "↑" if r['dii_score'] - r['dii_score_prev'] > 0.02
                    else "↓" if r['dii_score'] - r['dii_score_prev'] < -0.02
                    else "→",
        axis=1
    )

    df.to_csv("data/pond_data.csv", index=False)
    print(df[['pond_name','dii_score','risk_category']])


if __name__ == "__main__":
    run()