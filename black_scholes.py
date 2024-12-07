import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price
    
    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    
    Returns:
    call_price: Call option price
    """
    if T == 0 or sigma == 0:
        return max(0, S - K)
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes put option price using put-call parity
    
    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    
    Returns:
    put_price: Put option price
    """
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = call_price + K*np.exp(-r*T) - S
    return put_price

def calculate_greeks(S, K, T, r, sigma):
    """
    Calculate option Greeks
    
    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    
    Returns:
    dict: Dictionary containing all Greeks
    """
    if T == 0 or sigma == 0:
        return {
            'delta_call': 1.0 if S > K else 0.0,
            'delta_put': -1.0 if S > K else 0.0,
            'gamma': 0.0,
            'theta_call': 0.0,
            'theta_put': 0.0,
            'vega': 0.0,
            'rho_call': 0.0,
            'rho_put': 0.0
        }
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Calculate Greeks
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    
    theta_call = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                 r*K*np.exp(-r*T)*norm.cdf(d2))
    theta_put = theta_call + r*K*np.exp(-r*T)
    
    vega = S*np.sqrt(T)*norm.pdf(d1)
    
    rho_call = K*T*np.exp(-r*T)*norm.cdf(d2)
    rho_put = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    
    return {
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'theta_call': theta_call/365,  # Convert to daily theta
        'theta_put': theta_put/365,    # Convert to daily theta
        'vega': vega/100,              # Convert to 1% vol change
        'rho_call': rho_call/100,      # Convert to 1% rate change
        'rho_put': rho_put/100         # Convert to 1% rate change
    } 