import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
raw_data = """event_id,plant_name,region,season,ndwi_score,ndvi_score,slope_degrees,rainfall_forecast_mm,breach_proximity_score,observation_duration_days,breached,failure_mode,source
E001,Sasan UMPP,Singrauli,monsoon,0.46,0.08,21,142,0.85,14,1,seepage_failure,NGT_Hira_Lal_Bais
E002,Essar Mahan Power,Singrauli,monsoon,0.38,0.12,18,158,0.78,9,1,monsoon_saturation,Manthan_2021
E003,NTPC Vindhyachal,Singrauli,post_monsoon,0.32,0.18,16,88,0.72,21,1,structural_deficiency,NGT_OA_164_2018
E004,NTPC Korba,Korba,monsoon,0.41,0.10,19,134,0.80,12,1,seepage_failure,HEI_Vol2_2021
E005,NTPC Korba,Korba,monsoon,0.35,0.15,19,122,0.75,18,1,monsoon_saturation,SANDRP_2022
E006,BSPCL Bokaro,Bokaro,monsoon,0.39,0.11,17,148,0.70,16,1,overtopping,SANDRP_2022
E007,BSPCL Bokaro,Bokaro,post_monsoon,0.28,0.22,17,96,0.68,22,1,drainage_blockage,SANDRP_2022
E008,Khaparkheda TPS,Nagpur,monsoon,0.42,0.09,15,161,0.74,11,1,overtopping,SANDRP_2022
E009,Koradi TPS,Nagpur,monsoon,0.44,0.08,14,155,0.76,6,1,structural_deficiency,SANDRP_2022
E010,ITPS Jharsuguda,Jharsuguda,post_monsoon,0.31,0.20,13,72,0.65,25,1,drainage_blockage,SANDRP_2023
E011,NTPC Talcher,Talcher,monsoon,0.36,0.14,16,138,0.71,17,1,monsoon_saturation,HEI_Compendium_2020
E012,North Chennai PS,Chennai,monsoon,0.40,0.11,12,145,0.68,13,1,seepage_failure,HEI_Compendium_2020
E013,Sasan UMPP,Singrauli,winter,0.09,0.58,21,14,0.85,90,0,none,NGT_Hira_Lal_Bais
E014,Sasan UMPP,Singrauli,pre_monsoon,0.19,0.34,21,52,0.85,45,0,none,NGT_Hira_Lal_Bais
E015,Essar Mahan Power,Singrauli,winter,0.07,0.61,18,11,0.78,90,0,none,Manthan_2021
E016,Essar Mahan Power,Singrauli,pre_monsoon,0.17,0.38,18,44,0.78,45,0,none,Manthan_2021
E017,NTPC Vindhyachal,Singrauli,winter,0.06,0.63,16,12,0.72,90,0,none,NGT_OA_164_2018
E018,NTPC Vindhyachal,Singrauli,monsoon,0.24,0.26,16,98,0.72,38,0,none,NGT_OA_164_2018
E019,NTPC Korba,Korba,winter,0.08,0.56,19,13,0.80,90,0,none,HEI_Vol2_2021
E020,NTPC Korba,Korba,pre_monsoon,0.18,0.36,19,48,0.80,45,0,none,HEI_Vol2_2021
E021,BSPCL Bokaro,Bokaro,winter,0.07,0.60,17,10,0.70,90,0,none,SANDRP_2022
E022,BSPCL Bokaro,Bokaro,pre_monsoon,0.16,0.40,17,38,0.70,45,0,none,SANDRP_2022
E023,Khaparkheda TPS,Nagpur,winter,0.08,0.55,15,12,0.74,90,0,none,SANDRP_2022
E024,Khaparkheda TPS,Nagpur,pre_monsoon,0.18,0.35,15,46,0.74,45,0,none,SANDRP_2022
E025,Koradi TPS,Nagpur,winter,0.07,0.57,14,11,0.76,90,0,none,SANDRP_2022
E026,Koradi TPS,Nagpur,pre_monsoon,0.17,0.37,14,43,0.76,45,0,none,SANDRP_2022
E027,ITPS Jharsuguda,Jharsuguda,winter,0.08,0.59,13,13,0.65,90,0,none,SANDRP_2023
E028,ITPS Jharsuguda,Jharsuguda,pre_monsoon,0.16,0.42,13,40,0.65,45,0,none,SANDRP_2023
E029,NTPC Talcher,Talcher,winter,0.07,0.62,16,11,0.71,90,0,none,HEI_Compendium_2020
E030,NTPC Talcher,Talcher,pre_monsoon,0.15,0.41,16,42,0.71,45,0,none,HEI_Compendium_2020
E031,North Chennai PS,Chennai,winter,0.06,0.64,12,18,0.68,90,0,none,HEI_Compendium_2020
E032,North Chennai PS,Chennai,pre_monsoon,0.14,0.44,12,55,0.68,45,0,none,HEI_Compendium_2020
E033,Rihand STPS,Singrauli,monsoon,0.26,0.24,11,95,0.35,35,0,none,HEI_Compendium_2020
E034,Rihand STPS,Singrauli,winter,0.07,0.60,11,12,0.35,90,0,none,HEI_Compendium_2020
E035,CSEB Korba West,Korba,monsoon,0.18,0.38,6,88,0.30,42,0,none,DownToEarth_2021
E036,CSEB Korba West,Korba,winter,0.06,0.63,6,10,0.30,90,0,none,DownToEarth_2021
E037,Korba East TPS,Korba,monsoon,0.14,0.44,5,72,0.25,42,0,none,DownToEarth_2021
E038,Korba East TPS,Korba,winter,0.05,0.66,5,9,0.25,90,0,none,DownToEarth_2021
E039,Hasdeo TPS,Korba,monsoon,0.12,0.52,7,68,0.20,42,0,none,DownToEarth_2021
E040,Hasdeo TPS,Korba,winter,0.05,0.68,7,8,0.20,90,0,none,DownToEarth_2021
E041,NTPC Vindhyachal,Singrauli,post_monsoon,0.14,0.46,16,42,0.72,60,0,none,NGT_OA_164_2018
E042,NTPC Vindhyachal,Singrauli,pre_monsoon,0.16,0.40,16,48,0.72,45,0,none,NGT_OA_164_2018
E043,Sasan UMPP,Singrauli,post_monsoon,0.28,0.22,21,78,0.85,28,0,none,NGT_Hira_Lal_Bais
E044,NTPC Korba,Korba,post_monsoon,0.26,0.24,19,82,0.80,30,0,none,HEI_Vol2_2021
E045,Koradi TPS,Nagpur,post_monsoon,0.24,0.28,14,76,0.76,32,0,none,SANDRP_2022
E046,Rihand STPS,Singrauli,post_monsoon,0.12,0.48,11,38,0.35,60,0,none,HEI_Compendium_2020
E047,BSPCL Bokaro,Bokaro,pre_monsoon,0.15,0.42,17,36,0.70,45,0,none,SANDRP_2022
E048,NTPC Talcher,Talcher,post_monsoon,0.14,0.45,16,44,0.71,50,0,none,HEI_Compendium_2020"""

df = pd.read_csv(io.StringIO(raw_data))

# Isolate structural/environmental predictor components
feature_cols = [
    'ndwi_score', 'ndvi_score', 'slope_degrees', 
    'rainfall_forecast_mm', 'breach_proximity_score', 'observation_duration_days'
]
X = df[feature_cols]

# Standardize to zero mean and unit variance (Crucial for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. RUN DECOMPOSITION (PCA)
# ==========================================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Determine Explained Variance Ratio
var_explained = pca.explained_variance_ratio_
cum_var_explained = np.cumsum(var_explained)

print("--- PCA STATISTICAL MATRIX SUMMARY ---")
for i, var in enumerate(var_explained):
    print(f"Principal Component {i+1}: {var*100:.2f}% Variance Captured (Cumulative: {cum_var_explained[i]*100:.2f}%)")

# ==========================================
# 3. EXTRACT FEATURE IMPORTANCE (LOADINGS)
# ==========================================
# Loadings are the coefficients of the linear combination of the original variables
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(len(feature_cols))], 
    index=feature_cols
)

print("\n--- COMPONENT LOADINGS MATRIX (WEIGHTS) ---")
print(loadings[['PC1', 'PC2']].round(3))

# Calculate absolute cumulative importance over top components
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'PC1_Absolute_Weight': np.abs(loadings['PC1']),
    'Total_Variance_Contribution': np.sum(np.abs(loadings.iloc[:, :2]), axis=1)
}).sort_values(by='Total_Variance_Contribution', ascending=False)

print("\n--- DERIVED RANKING OF KEY DRIVERS ---")
print(feature_importance.to_string(index=False))

# ==========================================
# 4. RENDER GRAPHICAL INFERENCE PLOTS
# ==========================================
plt.figure(figsize=(15, 5))

# Plot A: Scree Plot (Variance Profile)
plt.subplot(1, 3, 1)
plt.bar(range(1, len(var_explained)+1), var_explained*100, alpha=0.7, color='darkblue', label='Individual')
plt.step(range(1, len(cum_var_explained)+1), cum_var_explained*100, where='mid', color='red', marker='o', label='Cumulative')
plt.title('Scree Variance Profile Plot')
plt.xlabel('Principal Component Index')
plt.ylabel('Variance Contribution Percentage (%)')
plt.grid(True, linestyle='--')
plt.legend()

# Plot B: Loading Heatmap (Feature Importance Profiles)
plt.subplot(1, 3, 2)
sns.heatmap(loadings[['PC1', 'PC2']], annot=True, cmap='RdBu_r', vmin=-1, vmax=1, cbar=True)
plt.title('Feature Coordinate Weights Heatmap')
plt.ylabel('Sensor Feature Track')

# Plot C: 2D Principal Component Projection Space
plt.subplot(1, 3, 3)
colors = {0: 'forestgreen', 1: 'crimson'}
labels = {0: 'Operational Stable', 1: 'Breached Containment'}
for breached_status in [0, 1]:
    mask = df['breached'] == breached_status
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1], 
        c=colors[breached_status], label=labels[breached_status],
        edgecolors='black', s=60, alpha=0.85
    )
plt.title('2D PCA Cluster Separation Space')
plt.xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
plt.grid(True, linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()