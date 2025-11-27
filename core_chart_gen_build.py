# Install dependencies:
# pip install yfinance pandas numpy scipy matplotlib seaborn

import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
TICKER = "CVX"          # Ticker to analyze
RISK_FREE_RATE = 0.045  # Approx. 4.5% risk-free rate
MIN_OI = 100            # Minimum Open Interest to filter out illiquid strikes
SMOOTHING_FACTOR = 0.5  # Controls how loose/tight the spline fit is (0-5 range usually)

# Time Frame Selection
# Options: "Short", "Medium", "Long", "All"
EXPIRATION_MODE = "Medium"

# Display Settings
MAX_EXPIRATIONS = 30    # Max number of expiration dates to show

# AI / Analysis Mode
# Enables explicit grid lines, text annotations for probabilities, 
# target prices, and expected moves.
AI_READABILITY_MODE = True 

def black_scholes_greeks(S, K, T, r, sigma, opt_type='call'):
    """
    Calculates Black-Scholes Greeks for a single option.
    Returns a formatted string of Greeks.
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    delta = si.norm.cdf(d1) if opt_type == 'call' else si.norm.cdf(d1) - 1
    
    # Gamma (Same for Call/Put)
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta (Annualized, usually divided by 365 for daily view)
    theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    theta = theta / 365.0 # Daily theta

    return {"delta": delta, "gamma": gamma, "theta": theta}

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Standard Black-Scholes Call Price formula.
    """
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

def get_earnings_date(ticker_obj):
    """
    Fetches the next earnings date.
    """
    try:
        earnings = ticker_obj.earnings_dates
        if earnings is not None and not earnings.empty:
            future_earnings = earnings[earnings.index > datetime.now()]
            if not future_earnings.empty:
                return future_earnings.index[-1].strftime('%b %d')
    except Exception:
        pass
    return "N/A"

def get_volatility_stats(ticker_obj, current_iv):
    """
    Calculates Volatility Rank using Historical Volatility as a proxy
    since historical IV is not available for free.
    """
    try:
        hist = ticker_obj.history(period="1y")
        if hist.empty:
            return "N/A"
        
        # Calculate Log Returns
        hist['LogRet'] = np.log(hist['Close'] / hist['Close'].shift(1))
        # Calculate rolling 30d annualized volatility
        hist['Vol'] = hist['LogRet'].rolling(window=30).std() * np.sqrt(252)
        
        min_vol = hist['Vol'].min()
        max_vol = hist['Vol'].max()
        current_vol = hist['Vol'].iloc[-1]
        
        if max_vol != min_vol:
            rank = (current_iv - min_vol) / (max_vol - min_vol) * 100
            return f"{rank:.0f}%"
    except Exception:
        pass
    return "N/A"

def get_target_expirations(ticker_obj):
    """
    Filters available expirations based on the EXPIRATION_MODE configuration.
    """
    all_expirations = ticker_obj.options
    target_exps = []
    today = datetime.now()
    
    print(f"Filtering expirations for mode: {EXPIRATION_MODE}...")
    
    for date_str in all_expirations:
        try:
            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_to_exp = (exp_date - today).days
            
            if days_to_exp < 1:
                continue

            if EXPIRATION_MODE == "Short":
                if days_to_exp <= 45: target_exps.append(date_str)
            elif EXPIRATION_MODE == "Medium":
                if 30 <= days_to_exp <= 180: target_exps.append(date_str)
            elif EXPIRATION_MODE == "Long":
                if 365 <= days_to_exp <= 900: target_exps.append(date_str)
            else: 
                target_exps.append(date_str)
                
        except ValueError:
            continue
            
    if len(target_exps) > MAX_EXPIRATIONS:
        step = len(target_exps) // MAX_EXPIRATIONS + 1
        target_exps = target_exps[::step]
        print(f"  > Too many expirations found. Thinned to {len(target_exps)} dates.")
    else:
        print(f"  > Showing all {len(target_exps)} expirations found in range.")
        
    return target_exps

def get_option_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        hist = ticker.history(period="1d")
        if hist.empty:
            return None, None, None, None
        current_price = hist['Close'].iloc[-1]
    except Exception:
        return None, None, None, None

    target_expirations = get_target_expirations(ticker)
    
    if not target_expirations:
        print(f"No expirations found for mode {EXPIRATION_MODE}")
        return None, None, None, None
        
    all_pdfs = []
    all_ivs = [] 
    total_call_vol = 0
    total_put_vol = 0
    
    print(f"Processing data for {ticker_symbol} (Current Price: ${current_price:.2f})...")

    for exp_date_str in target_expirations:
        try:
            opt = ticker.option_chain(exp_date_str)
            calls = opt.calls
            puts = opt.puts
            
            # Aggregate Volume for PCR
            total_call_vol += calls['volume'].sum()
            total_put_vol += puts['volume'].sum()
            
            strike_range_pct = 0.4 if EXPIRATION_MODE in ["Long", "Medium"] else 0.2
            
            # Filter Calls
            calls = calls[
                (calls['openInterest'] > MIN_OI) & 
                (calls['strike'] > current_price * (1 - strike_range_pct)) & 
                (calls['strike'] < current_price * (1 + strike_range_pct))
            ].copy()
            
            if len(calls) < 10:
                continue

            all_ivs.extend(calls['impliedVolatility'].tolist())

            exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
            today = datetime.now()
            days_to_exp = (exp_date - today).days
            if days_to_exp <= 0:
                continue
            T = days_to_exp / 365.0

            # 1. Smooth IV & Calculate PDF
            calls = calls.sort_values('strike')
            strikes = calls['strike'].values
            ivs = calls['impliedVolatility'].values
            iv_spline = UnivariateSpline(strikes, ivs, k=3, s=SMOOTHING_FACTOR)
            
            dense_strikes = np.linspace(strikes.min(), strikes.max(), 200)
            smoothed_ivs = iv_spline(dense_strikes)
            
            smoothed_prices = np.array([
                black_scholes_call_price(current_price, k, T, RISK_FREE_RATE, iv) 
                for k, iv in zip(dense_strikes, smoothed_ivs)
            ])
            
            dk = dense_strikes[1] - dense_strikes[0]
            second_derivative = np.diff(smoothed_prices, 2) / (dk ** 2)
            pdf_strikes = dense_strikes[1:-1]
            pdf_values = second_derivative * np.exp(RISK_FREE_RATE * T)
            pdf_values = np.maximum(pdf_values, 0)

            # 2. Statistics (Expected Move)
            weighted_avg_price = current_price
            implied_std_dev = 0
            if np.sum(pdf_values) > 0:
                weighted_avg_price = np.average(pdf_strikes, weights=pdf_values)
                variance = np.average((pdf_strikes - weighted_avg_price)**2, weights=pdf_values)
                implied_std_dev = np.sqrt(variance)

            # 3. Calculate ATM Greeks for this Expiration (Snap shot of risk)
            # Find IV at current price
            atm_iv = float(iv_spline(current_price))
            greeks = black_scholes_greeks(current_price, current_price, T, RISK_FREE_RATE, atm_iv)

            exp_df = pd.DataFrame({
                'Strike': pdf_strikes,
                'Probability': pdf_values,
                'Expiration': exp_date_str,
                'DaysToExp': days_to_exp,
                'ExpectedPrice': weighted_avg_price,
                'ImpliedMove': implied_std_dev,
                'Delta': greeks['delta'],
                'Gamma': greeks['gamma'],
                'Theta': greeks['theta']
            })
            
            all_pdfs.append(exp_df)
            
        except Exception:
            continue

    if not all_pdfs:
        return None, None, None, None

    # Get Metadata
    avg_iv = np.mean(all_ivs) if all_ivs else 0
    earnings_date = get_earnings_date(ticker)
    vol_rank = get_volatility_stats(ticker, avg_iv)
    pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0

    meta = {
        "earnings": earnings_date, 
        "rank": vol_rank,
        "pcr": pcr
    }

    return pd.concat(all_pdfs), current_price, avg_iv, meta

def plot_heatmap(df, current_price, ticker, avg_iv, meta):
    # 1. Bins
    min_strike = df['Strike'].min()
    max_strike = df['Strike'].max()
    
    step = 5 if current_price > 200 else 1
    if EXPIRATION_MODE == "Long": step = 10 
        
    bins = np.arange(np.floor(min_strike), np.ceil(max_strike), step)
    
    df['StrikeBin'] = pd.cut(df['Strike'], bins=bins, labels=bins[:-1])
    
    # 2. Aggregation
    heatmap_data = df.groupby(['Expiration', 'StrikeBin'])['Probability'].mean().reset_index()
    heatmap_matrix = heatmap_data.pivot(index='StrikeBin', columns='Expiration', values='Probability')
    
    # Sort
    heatmap_matrix.columns = pd.to_datetime(heatmap_matrix.columns)
    heatmap_matrix = heatmap_matrix.sort_index(axis=1) 
    heatmap_matrix.columns = heatmap_matrix.columns.strftime('%Y-%m-%d')
    heatmap_matrix = heatmap_matrix.sort_index(ascending=False)
    
    # 3. Plot Setup
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Adjust Margins: HUGE top margin for HUD + Greeks, bottom for Note
    plt.subplots_adjust(top=0.82, bottom=0.15) 
    
    # Zebra Striping
    y_ticks = np.arange(len(heatmap_matrix.index))
    for i in range(0, len(y_ticks), 2):
        ax.axhspan(i, i+1, color='white', alpha=0.05, zorder=2) 
        ax.axhspan(i+1, i+2, color='black', alpha=0.05, zorder=2)

    # Grid Lines
    heatmap_kws = {'linewidths': 0.05, 'linecolor': 'black'} if AI_READABILITY_MODE else {}

    # Draw Heatmap
    sns.heatmap(
        heatmap_matrix, 
        cmap="magma", 
        cbar_kws={'label': f'Probability Density (Relative)'},
        yticklabels=True,
        xticklabels=True,
        ax=ax,
        **heatmap_kws
    )
    
    # --- TRENDLINE & SENTIMENT ---
    price_map = df.groupby('Expiration')['ExpectedPrice'].first()
    trend_prices = []
    x_coords = []
    
    for i, date_col in enumerate(heatmap_matrix.columns):
        if date_col in price_map:
            trend_prices.append(price_map[date_col])
            x_coords.append(i + 0.5)

    strikes_y_axis = np.array([float(x) for x in heatmap_matrix.index])
    y_coords = []
    
    trend_label = "NEUTRAL"
    trend_color = 'cyan' 
    
    if len(trend_prices) > 1:
        if trend_prices[-1] > trend_prices[0]:
            trend_color = '#00ff00' # Green
            trend_label = "BULLISH"
        else:
            trend_color = '#ff3333' # Red
            trend_label = "BEARISH"

    if len(strikes_y_axis) > 1:
        strikes_asc = strikes_y_axis[::-1]
        indices_desc = np.arange(len(strikes_y_axis))[::-1]
        
        for price in trend_prices:
            y_idx = np.interp(price, strikes_asc, indices_desc)
            y_coords.append(y_idx + 0.5) 
            
        plt.plot(x_coords, y_coords, color=trend_color, marker='o', linestyle='-', linewidth=2.5, label=f'Consensus Trend ({trend_label})')

        # Annotation: Trend Endpoint (ALWAYS ON)
        if len(trend_prices) > 0:
            last_price = trend_prices[-1]
            plt.text(x_coords[-1] + 0.2, y_coords[-1], f"Proj: ${last_price:.2f}", 
                     color='white', fontweight='bold', fontsize=10, 
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor=trend_color))

    # --- TOP ANNOTATIONS (Greeks & Move) ---
    move_map = df.groupby('Expiration')['ImpliedMove'].first()
    delta_map = df.groupby('Expiration')['Delta'].first()
    gamma_map = df.groupby('Expiration')['Gamma'].first()
    theta_map = df.groupby('Expiration')['Theta'].first()

    for i, date_col in enumerate(heatmap_matrix.columns):
        if date_col in move_map:
            move = move_map[date_col]
            d = delta_map[date_col]
            g = gamma_map[date_col]
            t = theta_map[date_col]
            
            # Format: {Δ0.50 Γ0.02 Θ-0.10}
            greeks_str = f"{{Δ{d:.2f} Γ{g:.2f} Θ{t:.2f}}}"
            
            # Greeks Line (Higher up)
            ax.text(i + 0.5, -1.5, greeks_str, 
                     ha='left', va='bottom', color='#AAAAAA', fontsize=7, rotation=45, fontweight='bold')
            
            # Expected Move Line (Just above chart)
            ax.text(i + 0.5, -0.2, f"±${move:.1f}", 
                     ha='center', va='bottom', color='white', fontsize=8,
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0))

    # --- HOTSPOTS (Always On) ---
    for col_idx in range(heatmap_matrix.shape[1]):
        col_data = heatmap_matrix.iloc[:, col_idx]
        max_row_idx = col_data.argmax()
        max_val = col_data.iloc[max_row_idx]
        
        prob_mass = max_val * step
        
        if prob_mass > 0.01: 
            ax.text(col_idx + 0.5, max_row_idx + 0.5, f"{prob_mass:.0%}", 
                     ha='center', va='center', color='black', fontsize=7, fontweight='bold',
                     bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="none", alpha=0.6))

    # --- CURRENT PRICE LINE (Cyan) ---
    try:
        closest_strike = min(strikes_y_axis, key=lambda x: abs(x - current_price))
        y_pos = list(heatmap_matrix.index).index(closest_strike)
        ax.axhline(y=y_pos + 0.5, color='cyan', linestyle='--', linewidth=1.5, label=f'Spot: ${current_price:.2f}')
    except:
        pass

    # --- HUD (Head-Up Display) ---
    plt.text(0, 1.25, f"  {ticker} ANALYSIS HUD  ", transform=ax.transAxes, 
             fontsize=14, fontweight='bold', color='white', backgroundcolor='#333333')
    
    hud_line1 = (f"Price: ${current_price:.2f}  |  Trend: {trend_label}  |  "
                 f"Put/Call Ratio: {meta['pcr']:.2f}")
    
    hud_line2 = (f"Avg IV: {avg_iv*100:.1f}%  |  IV Rank (Proxy): {meta['rank']}  |  "
                 f"Next Earnings: {meta['earnings']}")
    
    plt.text(0, 1.20, hud_line1, transform=ax.transAxes, fontsize=11, color='black', fontweight='bold')
    plt.text(0, 1.16, hud_line2, transform=ax.transAxes, fontsize=11, color='black', fontweight='medium')

    # --- Y-Axis Bold Formatting ---
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        try:
            val = float(label.get_text())
            if val % 10 == 0: 
                label.set_fontweight('bold')
                label.set_fontsize(11)
        except:
            pass
    ax.set_yticklabels(y_labels)

    # --- Colorbar Modification ---
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.plot([0, 1], [0.5, 0.5], color='white', linewidth=2, transform=cbar.ax.transAxes)
        cbar.ax.text(1.1, 0.5, '50% Intensity', transform=cbar.ax.transAxes, 
                     va='center', fontsize=8, color='black')

    # --- AI Footer Note ---
    if AI_READABILITY_MODE:
        note = ("AI NOTE: Cyan Line = Spot Price. Green/Red Line = Bullish/Bearish Trend. "
                "Curly Brackets {} = ATM Greeks (Delta, Gamma, Theta). Hotspots = High Probability.")
        plt.figtext(0.5, 0.02, note, ha="center", fontsize=10, 
                    bbox={"facecolor":"cyan", "alpha":0.1, "pad":5})

    plt.legend(loc='upper right')
    plt.title(f"Market Implied Probability Heatmap: {ticker} ({EXPIRATION_MODE} Term)", fontsize=16, y=1.28)
    plt.xlabel("Expiration Date", fontsize=12)
    plt.ylabel("Strike Price ($)", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.show()

if __name__ == "__main__":
    print(f"Fetching Option Data for mode: {EXPIRATION_MODE}")
    pdf_data, spot_price, avg_iv, meta = get_option_data(TICKER)
    
    if pdf_data is not None:
        plot_heatmap(pdf_data, spot_price, TICKER, avg_iv, meta)
        print("Done. Heatmap generated.")
    else:
        print("Failed to generate data.")