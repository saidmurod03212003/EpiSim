import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def seirv_model(y, t, N, beta, sigma, gamma, nu):
    """
    SEIRV model equations including vaccination rate nu.
    """
    S, E, I, R, V = y
    dSdt = -beta * S * I / N - nu * S
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S
    return dSdt, dEdt, dIdt, dRdt, dVdt

def run_simulation(N, E0, I0, R0, V0, beta, sigma, gamma, nu, days):
    """
    Executes the SEIRV simulation.
    """
    S0 = N - E0 - I0 - R0 - V0
    y0 = S0, E0, I0, R0, V0
    t = np.linspace(0, days, int(days))
    
    ret = odeint(seirv_model, y0, t, args=(N, beta, sigma, gamma, nu))
    return t, ret.T

def fit_parameters(t_data, i_data, N, E0, I0, R0, V0, sigma, nu):
    """
    Fits beta and gamma parameters to observed infection data.
    """
    def objective(params):
        beta, gamma = params
        _, (_, _, I, _, _) = run_simulation(N, E0, I0, R0, V0, beta, sigma, gamma, nu, len(t_data))
        I_resampled = np.interp(t_data, np.arange(len(I)), I)
        return np.mean((I_resampled - i_data)**2)

    res = minimize(objective, [0.5, 0.1], bounds=[(0.01, 2.0), (0.01, 1.0)])
    return res.x 
