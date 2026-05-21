import numpy as np
import pandas as pd
import pickle
import os
import shap
import warnings
warnings.filterwarnings('ignore')

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
import config

MODEL_DIR    = config.MODEL_DIR
FEATURE_COLS = config.FEATURE_COLS

os.makedirs(MODEL_DIR, exist_ok=True)

def get_training_data():
    df = pd.read_csv("data/pond_training.csv")
    
    train_cols = FEATURE_COLS + ['observation_duration_days', 'breached']
    df_train   = df[train_cols].copy()

    np.random.seed(42)
    aug_rows = []
    for _, row in df_train.iterrows():
        for _ in range(20):
            noise = np.array([
                np.random.normal(0, 0.02),
                np.random.normal(0, 0.02),
                np.random.normal(0, 0.8),
                np.random.normal(0, 6.0),
                np.random.normal(0, 0.02),
            ])
            new_features = row[FEATURE_COLS].values + noise
            new_features = np.clip(new_features, 0, [0.55, 0.70, 25, 180, 1.0])
            aug_row = dict(zip(FEATURE_COLS, new_features))
            aug_row['observation_duration_days'] = row['observation_duration_days']
            aug_row['breached']                  = row['breached']
            aug_rows.append(aug_row)

    df_aug   = pd.DataFrame(aug_rows)
    df_final = pd.concat([df_train, df_aug], ignore_index=True)
    return df_final

def train_and_save():
    df_train = get_training_data()
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        df_train,
        duration_col='observation_duration_days',
        event_col='breached',
    )
    
    scaler = StandardScaler()
    scaler.fit(df_train[FEATURE_COLS].values)
    
    with open(os.path.join(MODEL_DIR, "cox_model.pkl"), "wb") as f:
        pickle.dump(cph, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    print("Cox PH model trained and saved.")
    print(cph.summary[['coef','exp(coef)','p']].round(3))
    return cph, scaler

def load_model():
    mp = os.path.join(MODEL_DIR, "logistic_model.pkl")
    sp = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(mp):
        return train_and_save()
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(sp, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def compute_breach_probability(row, model, scaler):
    X = pd.DataFrame([[
        row['ndwi_score'], row['ndvi_score'], row['slope_degrees'],
        row['rainfall_forecast_mm'], row['breach_proximity_score']
    ]], columns=FEATURE_COLS)
    X['observation_duration_days'] = 30
    X['breached']                  = 0
    survival_prob = model.predict_survival_function(X, times=[30]).values[0][0]
    return round(float(1 - survival_prob), 3)

def compute_shap_values(row, model, scaler):
    df_train = get_training_data()
    X_train = df_train[FEATURE_COLS].values
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), size=min(50, len(X_train)), replace=False)
    X_background = X_train[bg_idx]

    def predict_fn(X):
        df_temp = pd.DataFrame(X, columns=FEATURE_COLS)
        return model.predict_partial_hazard(df_temp).values

    explainer = shap.KernelExplainer(predict_fn, X_background)

    explainer = shap.KernelExplainer(predict_fn, X_background)
    X_pond = np.array([[
        row['ndwi_score'], row['ndvi_score'], row['slope_degrees'],
        row['rainfall_forecast_mm'], row['breach_proximity_score']
    ]])
    sv = explainer.shap_values(X_pond, nsamples=100)[0]

    return {
        'ndwi':       round(float(sv[0]), 4),
        'ndvi':       round(float(sv[1]), 4),
        'slope':      round(float(sv[2]), 4),
        'rainfall':   round(float(sv[3]), 4),
        'proximity':  round(float(sv[4]), 4),
        'base_value': round(float(explainer.expected_value), 4),
    }

def compute_bayesian_dii(row, n=1000, noise_std=0.02):
    scores = []
    for _ in range(n):
        ndwi = max(0, row['ndwi_score']  + np.random.normal(0, noise_std))
        ndvi = max(0, min(0.70, row['ndvi_score'] + np.random.normal(0, noise_std)))
        rain = max(0, row['rainfall_forecast_mm'] + np.random.normal(0, 5))
        sar_raw = row.get('sar_backscatter', 0.05)
        sar_raw = 0.05 if (sar_raw is None or pd.isna(sar_raw)) else sar_raw
        sar = max(0, sar_raw + np.random.normal(0, 0.01))
        dii  = (
            config.W_NDWI      * min(ndwi / config.NDWI_MAX, 1.0) +
            config.W_NDVI      * min((1 - ndvi) / config.NDVI_STRESS_MAX, 1.0) +
            config.W_SLOPE     * min(row['slope_degrees'] / config.SLOPE_MAX, 1.0) +
            config.W_RAINFALL  * min(rain / config.RAINFALL_MAX, 1.0) +
            config.W_PROXIMITY * row['breach_proximity_score'] +
            config.W_SAR       * min(sar / config.SAR_MAX, 1.0)
        )
        scores.append(dii)
    return (
        round(float(np.mean(scores)), 3),
        round(float(np.percentile(scores, 2.5)), 3),
        round(float(np.percentile(scores, 97.5)), 3),
    )

def compute_mahalanobis(df):
    features = df[FEATURE_COLS].values.astype(float)
    mean     = np.mean(features, axis=0)
    cov      = np.cov(features.T)
    inv_cov  = np.linalg.pinv(cov)
    distances = [mahalanobis(row, mean, inv_cov) for row in features]
    threshold = np.percentile(distances, 75)
    df['mahal_distance']  = [round(d, 3) for d in distances]
    df['profile_anomaly'] = df['mahal_distance'] > threshold
    return df

def run_all_ml(df):
    model, scaler = load_model()

    df['breach_probability'] = df.apply(
        lambda r: compute_breach_probability(r, model, scaler), axis=1
    )

    bayes = df.apply(lambda r: compute_bayesian_dii(r), axis=1, result_type='expand')
    df['dii_mean']  = bayes[0]
    df['dii_lower'] = bayes[1]
    df['dii_upper'] = bayes[2]
    df['dii_score'] = df['dii_mean']

    df = compute_mahalanobis(df)

    def categorise(dii):
        if dii > config.CRITICAL_THRESHOLD:   return "CRITICAL"
        elif dii > config.ELEVATED_THRESHOLD: return "ELEVATED"
        return "STABLE"

    df['risk_category'] = df['dii_score'].apply(categorise)
    df.to_csv("data/pond_data.csv", index=False)

    print(df[['pond_name','dii_score','dii_lower','dii_upper',
              'breach_probability','profile_anomaly','risk_category']])
    print(f"\nCRITICAL: {len(df[df.risk_category=='CRITICAL'])}")
    print(f"ELEVATED: {len(df[df.risk_category=='ELEVATED'])}")
    print(f"STABLE:   {len(df[df.risk_category=='STABLE'])}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/pond_data.csv")
    run_all_ml(df)