import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import os

# --- PATHS ---
raw_path = r"..\data\raw_data\databycountry\United_states.csv"
save_path = r"..\data\processed_data\United_states.csv"

# --- SELECTED FEATURES ---
selected_features = [
    "year",
    "coal_co2", "coal_co2_per_capita",
    "oil_co2", "oil_co2_per_capita",
    "gas_co2", "gas_co2_per_capita",
    "cement_co2", "cement_co2_per_capita",
    "land_use_change_co2", "land_use_change_co2_per_capita",
    "ghg_per_capita", "ghg_excluding_lucf_per_capita",
    "total_ghg", "total_ghg_excluding_lucf",
    "methane", "nitrous_oxide",
    "co2", "temperature_change_from_co2", "temperature_change_from_ghg"
]

# --- COLUMNS TO KEEP INTACT (no scaling) ---
no_scale_columns = ["co2", "total_ghg", "temperature_change_from_co2", "temperature_change_from_ghg"]

# --- PCA GROUPS ---
pca_groups = {
    "coal": ["coal_co2", "coal_co2_per_capita"],
    "oil": ["oil_co2", "oil_co2_per_capita"],
    "gas": ["gas_co2", "gas_co2_per_capita"],
    "cement": ["cement_co2", "cement_co2_per_capita"],
    "land_use": ["land_use_change_co2", "land_use_change_co2_per_capita"],
    "ghg": ["ghg_per_capita", "ghg_excluding_lucf_per_capita", "total_ghg_excluding_lucf"],
    "nonco2_ghg": ["methane", "nitrous_oxide"]
}

# --- LOAD DATA ---
df = pd.read_csv(raw_path)
df = df[selected_features].copy()

# --- INTERPOLATE MISSING VALUES ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'year']  # --- MODIFIED ---

# Function to impute using rolling window with ffill and bfill
def rolling_window_impute(df, cols, window_size=5):
    df_imputed = df.copy()

    for col in cols:
        if df[col].isna().sum() > 0:
            print(f"  - Imputing column: {col}")
            mask = df[col].isna()

            for idx in df.index[mask]:
                lower_bound = max(0, idx - window_size)
                upper_bound = min(len(df), idx + window_size + 1)
                window_df = df.iloc[lower_bound:upper_bound]

                temp_value = None
                forward_values = window_df.loc[lower_bound:idx-1, col].dropna()
                if len(forward_values) > 0:
                    temp_value = forward_values.iloc[-1]

                if temp_value is None:
                    backward_values = window_df.loc[idx+1:upper_bound, col].dropna()
                    if len(backward_values) > 0:
                        temp_value = backward_values.iloc[0]

                if temp_value is not None:
                    df_imputed.loc[idx, col] = temp_value

    # Final ffill/bfill fallback
    for col in cols:
        if df_imputed[col].isna().sum() > 0:
            df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')

    return df_imputed

# Apply rolling window imputation
print("Imputing missing values using rolling window with ffill/bfill method...")
df = rolling_window_impute(df, numeric_cols, window_size=5)

# --- PCA FUNCTION ---
def apply_pca(df, columns, n_components=1, name="pca"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    df[f"{name}_pc1"] = X_pca[:, 0]
    df.drop(columns=columns, inplace=True)
    return df

# --- APPLY PCA TO EACH GROUP ---
for name, cols in pca_groups.items():
    # --- MODIFY --- do not apply PCA if all columns in group are in no_scale_columns
    if not all(col in no_scale_columns for col in cols):
        df = apply_pca(df, cols, n_components=1, name=name)

# --- MINMAX SCALING ---
features_to_scale = [col for col in df.columns if col not in (['year'] + no_scale_columns)]  # --- MODIFIED ---
scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# --- SAVE PROCESSED FILE ---
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)

print(f"\u2705 Processed data saved to:\n{save_path}")