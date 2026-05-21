import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Select numerical engineering attributes and target outcome row
feature_cols = [
    'ndwi_score', 'ndvi_score', 'slope_degrees', 
    'rainfall_forecast_mm', 'breach_proximity_score', 
    'observation_duration_days', 'breached'
]
analysis_df = df[feature_cols]

# ==========================================
# 2. COMPUTE COVARIANCE MATRIX
# ==========================================
cov_matrix = analysis_df.cov()

print("--- RAW COVARIANCE MATRIX MATRIX ---")
print(cov_matrix.round(4))

# Isolate feature directional interaction mapping against the target variable 'breached'
target_covariance = cov_matrix['breached'].drop('breached')
sorted_drivers = target_covariance.abs().sort_values(ascending=False)

print("\n--- ATTRIBUTE DEVIATION RANKING RELATIVE TO BREACH OUTCOMES ---")
for idx in sorted_drivers.index:
    raw_val = target_covariance[idx]
    print(f"Feature: {idx:<26} | Covariance Vector: {raw_val:>8.4f}")

# ==========================================
# 3. GRAPHICAL HEATMAP GENERATION
# ==========================================
plt.figure(figsize=(10, 8))

# Mask the upper triangle to emphasize clean scannability
mask = np.triu(np.ones_like(cov_matrix, dtype=bool))

sns.heatmap(
    cov_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="PRGn",  # Diverging palette highlights positive vs negative variance steps
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Covariance Magnitude (Scale-Dependent)"}
)

plt.title("CoalWatch Engineering Feature Covariance Space Map", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()