import torch
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import functools
import hashlib
import torch.nn as nn

# Cache for loaded data files to avoid redundant disk I/O
_data_cache = {}

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # Input size is 1 since we only use past CO2 values
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        out = lstm_out[:, -1, :]
        
        # Apply dropout and fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out.squeeze()

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Forward pass through GRU
        gru_out, _ = self.gru(x)
        
        # Take the last time step output
        out = gru_out[:, -1, :]
        
        # Apply dropout and fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out.squeeze()

def get_data(data_path):
    """Load data from CSV file with caching"""
    if data_path not in _data_cache:
        _data_cache[data_path] = pd.read_csv(data_path)
    return _data_cache[data_path].copy()  # Return a copy to avoid modifying cached data

# Define model loading with caching
# Cache for loaded models
_model_cache = {}

def load_model(model_choice, emission_choice, region_choice):
    """Load machine learning model with caching to avoid redundant disk I/O"""

    
    # Create a cache key from the parameters
    cache_key = f"{model_choice}_{emission_choice}_{region_choice}"
    
    # If model is already in cache, return it
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Base model path follows the structure provided
    model_base_path = "models"
    
    # Replace space with underscore for directory names
    region_dir = region_choice.replace(" ", "_")
    
    # Complete model directory path
    model_dir = os.path.join(model_base_path, region_dir,emission_choice, model_choice)
    
    # List files in the model directory to get the correct file
    try:
        model_files = os.listdir(model_dir)
        
        # Filter for appropriate file extensions based on model type
        if model_choice in ["ARIMA", "ARIMA+LSTM"]:
            # Look for .pkl files for ARIMA models
            model_files = [f for f in model_files if f.endswith('.pkl')]
        elif model_choice in ["LSTM", "GRU"]:
            # Look for .pth files for neural network models
            model_files = [f for f in model_files if f.endswith('.pth')]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Take the first model file (since you mentioned there's only one in each folder)
        model_file = model_files[0]
        model_path = os.path.join(model_dir, model_file)
        
        # Load the model based on its type
        if model_choice in ["ARIMA", "ARIMA+LSTM"]:
            model = joblib.load(model_path)
        elif model_choice in ["LSTM", "GRU"]:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            raise ValueError(f"Model choice {model_choice} not recognized")
        
        # Store model in cache before returning
        _model_cache[cache_key] = model
        return model
    
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Forecasting future function with result caching
# Cache for forecast results
_forecast_cache = {}

def find_project_data_directory():
    """Find the data directory relative to the current file location"""
    # Get the directory of the current script (utils.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    # Construct the path to the data directory
    data_dir = os.path.join(current_dir, "data", "processed_data")
    
    # Print for debugging

    print(f"Project root directory: {current_dir}")
    print(f"Data directory: {data_dir}")
    
    return data_dir

def forecast_future(model_choice, model_data, forecast_years, emission_choice, region_choice):
    """Generate future forecasts with caching to avoid redundant computations"""
    # Create a cache key from the parameters
    cache_key = f"{model_choice}_{emission_choice}_{region_choice}_{forecast_years}"
    
    # If forecast is already in cache, return it
    if cache_key in _forecast_cache:
        return _forecast_cache[cache_key]
    
    
    # Find the data directory relative to where the script is running
    data_dir = find_project_data_directory()
    
    # Set the file name based on region
    if region_choice == "United states":
        file_name = "us.csv"
    else:
        file_name = "World.csv"
    
    data_path = os.path.join(data_dir, file_name)
    print(f"Attempting to load data from: {os.path.abspath(data_path)}")
    try:
        # Use the caching function to get data
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # Try alternative path format if the first one fails
        data_path = os.path.join("data","processed_data",file_name)
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find data file for {region_choice}. Tried paths: {os.path.abspath(data_path)} "
                               f"..\\data\\processed_data\\{region_choice.replace(' ', '_')}.csv and "
                               f"../data/processed_data/{region_choice.replace(' ', '_')}.csv")

    last_year = df['year'].max()
    future_years = np.arange(last_year + 1, last_year + forecast_years + 1)

    metrics = {}
    # Define columns based on emission choice
    target_column = 'co2' if emission_choice == "CO2" else 'total_ghg'
    log_target_column = 'log_co2' if emission_choice == "CO2" else 'log_ghg'
    
    # Make sure log columns exist, create them if they don't
    if log_target_column not in df.columns:
        df[log_target_column] = np.log1p(df[target_column])
    
    if model_choice == "ARIMA":
        # Full ARIMA model on log-transformed values
        arima_model = model_data
        # Use the log-transformed column for forecasting
        forecast = arima_model.forecast(steps=forecast_years)
        
        # Convert forecasts back to original scale
        forecast_original = np.expm1(forecast)

        # Create a dataframe with forecasts
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Forecasted_log_value': forecast,
            'Forecasted_CO2': forecast_original
        })
        
        # Calculate percentage changes
        forecast_df['pct_change'] = forecast_df['Forecasted_CO2'].pct_change() * 100
        # Use the original (non-log) column for comparison
        last_historical_value = df[target_column].iloc[-1]
        forecast_df['pct_change_from_last_historical'] = ((forecast_df['Forecasted_CO2'] - last_historical_value) / last_historical_value) * 100

        # Calculate confidence intervals (if available)
        try:
            conf_int = arima_model.get_forecast(steps=forecast_years).conf_int()
            conf_int_original = np.expm1(conf_int)
            forecast_df['lower_ci'] = conf_int_original.iloc[:, 0]
            forecast_df['upper_ci'] = conf_int_original.iloc[:, 1]
            
            # Add confidence intervals to the plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['year'], y=df[target_column], mode='lines', name='Historical Emissions'))
            fig.add_trace(go.Scatter(x=future_years, y=forecast_df['Forecasted_CO2'], mode='lines+markers', name='Forecasted Emissions'))
            fig.add_trace(go.Scatter(x=future_years, y=forecast_df['lower_ci'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=future_years, y=forecast_df['upper_ci'], mode='lines', line=dict(width=0), 
                                     fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', name='95% Confidence Interval'))
        except:
            # If confidence intervals are not available, create a plot without them
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['year'], y=df[target_column], mode='lines', name='Historical Emissions'))
            fig.add_trace(go.Scatter(x=future_years, y=forecast_df['Forecasted_CO2'], mode='lines+markers', name='Forecasted Emissions'))
        
        # Compute metrics
        metrics['avg_annual_pct_change'] = forecast_df['pct_change'].mean()
        metrics['total_pct_change'] = forecast_df['pct_change_from_last_historical'].iloc[-1]
        metrics['min_annual_pct_change'] = forecast_df['pct_change'].min()
        metrics['max_annual_pct_change'] = forecast_df['pct_change'].max()

    elif model_choice == "ARIMA+LSTM":
        # For ARIMA + LSTM, we'll use only the ARIMA part for now
        # since the hybrid implementation would require both models
        arima_model = model_data
        # Use the log-transformed column for forecasting
        arima_future = arima_model.forecast(steps=forecast_years)
        
        # Convert to original scale
        forecast_original = np.expm1(arima_future)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Forecasted_log_value': arima_future,
            'Forecasted_CO2': forecast_original
        })
        
        # Calculate percentage changes
        forecast_df['pct_change'] = forecast_df['Forecasted_CO2'].pct_change() * 100
        # Use the original (non-log) column for comparison
        last_historical_value = df[target_column].iloc[-1]
        forecast_df['pct_change_from_last_historical'] = ((forecast_df['Forecasted_CO2'] - last_historical_value) / last_historical_value) * 100
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['year'], y=df[target_column], mode='lines', name='Historical Emissions'))
        fig.add_trace(go.Scatter(x=future_years, y=forecast_df['Forecasted_CO2'], mode='lines+markers', name='Forecasted Emissions'))
        
        # Compute metrics
        metrics['avg_annual_pct_change'] = forecast_df['pct_change'].mean()
        metrics['total_pct_change'] = forecast_df['pct_change_from_last_historical'].iloc[-1]
        metrics['min_annual_pct_change'] = forecast_df['pct_change'].min()
        metrics['max_annual_pct_change'] = forecast_df['pct_change'].max()

    else:  # LSTM or GRU model
        # Prepare data for LSTM/GRU forecasting - use only the original (non-log) column
        # LSTM models were trained on normalized values of co2 or total_ghg
        feature_data = df[[target_column]]
        
        # Apply MinMaxScaler exactly as was done during training
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(feature_data)
        
        # Use the same sequence length as during training
        sequence_length = 10  # Using the same value as in your paste-2.txt
        X = []
        for i in range(len(normalized_data) - sequence_length):
            X.append(normalized_data[i:i+sequence_length])
        X = np.array(X)
        
        # Get the model and prepare for forecasting
        model = model_data
        model.eval()
        
        # Generate forecasts
        forecasted = []
        with torch.no_grad():
            current_input = torch.tensor(X[-1:], dtype=torch.float32)
            
            for _ in range(forecast_years):
                # Predict next value
                pred = model(current_input)
                forecasted.append(pred.item())
                
                # Update input sequence for next prediction
                current_input = torch.cat((
                    current_input[:, 1:], 
                    pred.view(1, 1, 1)
                ), dim=1)
        
        # Convert normalized predictions to original scale
        forecasted_array = np.array(forecasted).reshape(-1, 1)
        forecast_original = scaler.inverse_transform(forecasted_array).flatten()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'Forecasted_CO2': forecast_original
        })
        
        # Calculate percentage changes
        forecast_df['pct_change'] = forecast_df['Forecasted_CO2'].pct_change() * 100
        last_historical_value = df[target_column].iloc[-1]
        forecast_df['pct_change_from_last_historical'] = ((forecast_df['Forecasted_CO2'] - last_historical_value) / last_historical_value) * 100
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['year'], y=df[target_column], mode='lines', name='Historical Emissions'))
        fig.add_trace(go.Scatter(x=future_years, y=forecast_df['Forecasted_CO2'], mode='lines+markers', name='Forecasted Emissions'))
        
        # Compute metrics
        metrics['avg_annual_pct_change'] = forecast_df['pct_change'].dropna().mean()
        metrics['total_pct_change'] = forecast_df['pct_change_from_last_historical'].iloc[-1]
        metrics['min_annual_pct_change'] = forecast_df['pct_change'].dropna().min()
        metrics['max_annual_pct_change'] = forecast_df['pct_change'].dropna().max()
    
    # Finalize plot formatting
    fig.update_layout(
        title=f"{emission_choice} Emissions Forecast for {region_choice}",
        xaxis_title="Year",
        yaxis_title=f"{emission_choice} Emissions",
        legend_title="Data Source",
        template="plotly_white"
    )
    
    # Add a vertical line to separate historical from forecast
    fig.add_vline(x=last_year, line_dash="dash", line_color="gray")
    fig.add_annotation(x=last_year-2, y=df[target_column].max()*0.9,
                      text="Historical", showarrow=False, xanchor="right")
    fig.add_annotation(x=last_year+2, y=df[target_column].max()*0.9,
                      text="Forecast", showarrow=False, xanchor="left")
    
    # Store results in cache
    forecast_result = (forecast_df, fig, metrics)
    _forecast_cache[cache_key] = forecast_result
    
    return forecast_result
