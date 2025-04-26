import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import os

# --- PATHS ---
raw_path = r"C:\Users\acksh\OneDrive\Desktop\MSAI\Machine Learning\Final_project\data\raw_data\databycountry\United_States.csv"
save_path = r"C:\Users\acksh\OneDrive\Desktop\MSAI\Machine Learning\Final_project\data\processed_data\United_States.csv"

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
    "co2"
]

# --- PCA GROUPS ---
pca_groups = {
    "coal": ["coal_co2", "coal_co2_per_capita"],
    "oil": ["oil_co2", "oil_co2_per_capita"],
    "gas": ["gas_co2", "gas_co2_per_capita"],
    "cement": ["cement_co2", "cement_co2_per_capita"],
    "land_use": ["land_use_change_co2", "land_use_change_co2_per_capita"],
    "ghg": ["ghg_per_capita", "ghg_excluding_lucf_per_capita", "total_ghg", "total_ghg_excluding_lucf"],
    "nonco2_ghg": ["methane", "nitrous_oxide"]
}

# --- LOAD DATA ---
df = pd.read_csv(raw_path)
df = df[selected_features].copy()

# --- INTERPOLATE MISSING VALUES ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if (col != 'year' and col!="co2")]  # Exclude 'year' from imputation

# Function to impute using rolling window with ffill and bfill
def rolling_window_impute(df, cols, window_size=5):
    df_imputed = df.copy()
    
    for col in cols:
        # Check if column has missing values
        if df[col].isna().sum() > 0:
            print(f"  - Imputing column: {col}")
            
            # Use rolling mean with window_size for each missing value
            # We'll create a mask for NaN values
            mask = df[col].isna()
            
            # Apply rolling window
            for idx in df.index[mask]:
                # Define window boundaries
                lower_bound = max(0, idx - window_size)
                upper_bound = min(len(df), idx + window_size + 1)
                
                # Get window values
                window_df = df.iloc[lower_bound:upper_bound]
                
                # Use forward fill first
                temp_value = None
                forward_values = window_df.loc[lower_bound:idx-1, col].dropna()
                if len(forward_values) > 0:
                    temp_value = forward_values.iloc[-1]  # Get the nearest forward value
                
                # If no forward value, try backward fill
                if temp_value is None:
                    backward_values = window_df.loc[idx+1:upper_bound, col].dropna()
                    if len(backward_values) > 0:
                        temp_value = backward_values.iloc[0]  # Get the nearest backward value
                
                # If we found a value, use it
                if temp_value is not None:
                    df_imputed.loc[idx, col] = temp_value
    
    # For any remaining NaN values, use conventional ffill followed by bfill
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
    df = apply_pca(df, cols, n_components=1, name=name)


# --- MINMAX SCALING (EXCLUDING 'year' and 'co2') ---
features_to_scale = df.columns.drop(["year", "co2"])
scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# --- SAVE PROCESSED FILE ---
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)

print(f"✅ Processed data saved to:\n{save_path}")
