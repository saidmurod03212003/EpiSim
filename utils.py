import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_advanced_metrics(t, S, I, beta, gamma, threshold=10):
    """
    Computes R0, total infected, and epidemic duration.
    """
    # Basic Reproductive Number
    r0 = beta / gamma if gamma > 0 else 0
    
    # Total ever infected (those who left Susceptible state)
    total_ever_infected = S[0] - S[-1]
    
    # Duration when Infected > threshold
    active_indices = np.where(I > threshold)[0]
    if len(active_indices) > 1:
        duration = t[active_indices[-1]] - t[active_indices[0]]
    else:
        duration = 0
        
    return r0, total_ever_infected, duration

def generate_forecast(t, I, forecast_days=30):
    """
    Predicts future infections using linear regression on recent trend.
    """
    if len(I) < 10:
        return np.array([]), np.array([])
        
    # Use last 14 days for trend
    window = 14
    X = t[-window:].reshape(-1, 1)
    y = I[-window:]
    
    model = LinearRegression().fit(X, y)
    
    t_future = np.arange(t[-1] + 1, t[-1] + forecast_days + 1)
    I_future = model.predict(t_future.reshape(-1, 1))
    I_future = np.maximum(I_future, 0) # No negative infections
    
    return t_future, I_future

def validate_data_columns(df):
    """
    Checks if uploaded CSV has required 'Day' and 'Infected' columns.
    """
    required = {'Day', 'Infected'}
    return required.issubset(df.columns)
