# ------------------------------------------------------------------
# OPTION ANALYSIS DASHBOARD (Single File)
# ------------------------------------------------------------------
# A complete AI-powered Option Probability & Greeks visualization tool.
#
# RUN INSTRUCTIONS:
# 1. pip install streamlit yfinance pandas numpy scipy matplotlib seaborn
# 2. streamlit run dashboard.py
# ------------------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==========================================
#  CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Option Probability Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "Portal" look
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    h1, h2, h3 {
        color: #4da6ff !important;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
#  MATH ENGINE
# ==========================================

RISK_FREE_RATE = 0.045
MIN_OI = 100

def black_scholes_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2))

def black_scholes_greeks(S, K, T, r, sigma, opt_type='call'):
    if T <= 0 or sigma <= 0: return {"delta": 0, "gamma": 0, "theta": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = si.norm.cdf(d1) if opt_type == 'call' else si.norm.cdf(d1) - 1
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365.0
    return {"delta": delta, "gamma": gamma, "theta": theta}

@st.cache_data(ttl=3600) # Cache data for 1 hour to speed up interaction
def get_market_data(ticker_symbol, mode, smoothing):
    ticker = yf.Ticker(ticker_symbol)
    try:
        hist = ticker.history(period="1y")
        if hist.empty: return None, None, None
        current_price = hist['Close'].iloc[-1]
        
        # Volatility Rank Logic
        hist['LogRet'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['Vol'] = hist['LogRet'].rolling(window=30).std() * np.sqrt(252)
        min_v, max_v = hist['Vol'].min(), hist['Vol'].max()
        current_vol = hist['Vol'].iloc[-1]
        vol_rank = "N/A"
        if max_v != min_v:
            vol_rank = f"{((current_vol - min_v) / (max_v - min_v) * 100):.0f}%"

        # Earnings
        earnings_date = "N/A"
        try:
            earn = ticker.earnings_dates
            if earn is not None:
                future = earn[earn.index > datetime.now()]
                if not future.empty: earnings_date = future.index[-1].strftime('%b %d')
        except: pass

    except: return None, None, None

    # Option Chain Processing
    all_exps = ticker.options
    target_exps = []
    today = datetime.now()
    
    for date_str in all_exps:
        try:
            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
            days = (exp_date - today).days
            if days < 1: continue
            
            if mode == "Short" and days <= 45: target_exps.append(date_str)
            elif mode == "Medium" and 30 <= days <= 180: target_exps.append(date_str)
            elif mode == "Long" and 365 <= days <= 900: target_exps.append(date_str)
            elif mode == "All": target_exps.append(date_str)
        except: continue
        
    if len(target_exps) > 30:
        target_exps = target_exps[::(len(target_exps)//30 + 1)]

    if not target_exps: return None, None, None

    all_pdfs = []
    all_ivs = []
    total_call_vol = 0
    total_put_vol = 0

    progress_bar = st.progress(0)
    for i, exp_str in enumerate(target_exps):
        progress_bar.progress((i + 1) / len(target_exps))
        try:
            opt = ticker.option_chain(exp_str)
            calls, puts = opt.calls, opt.puts
            total_call_vol += calls['volume'].sum()
            total_put_vol += puts['volume'].sum()

            pct = 0.4 if mode in ["Long", "Medium"] else 0.2
            calls = calls[(calls['openInterest'] > MIN_OI) & 
                          (calls['strike'] > current_price * (1-pct)) & 
                          (calls['strike'] < current_price * (1+pct))].copy()
            if len(calls) < 10: continue
            
            all_ivs.extend(calls['impliedVolatility'].tolist())
            
            calls = calls.sort_values('strike')
            strikes = calls['strike'].values
            ivs = calls['impliedVolatility'].values
            
            # Spline
            iv_spline = UnivariateSpline(strikes, ivs, k=3, s=float(smoothing))
            
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - today).days / 365.0
            if T <= 0: continue

            dense_strikes = np.linspace(strikes.min(), strikes.max(), 200)
            smoothed_ivs = iv_spline(dense_strikes)
            prices = np.array([black_scholes_call_price(current_price, k, T, RISK_FREE_RATE, iv) for k, iv in zip(dense_strikes, smoothed_ivs)])
            
            dk = dense_strikes[1] - dense_strikes[0]
            # PDF Calculation
            pdf = np.maximum(np.diff(prices, 2) / (dk**2) * np.exp(RISK_FREE_RATE * T), 0)
            pdf_strikes = dense_strikes[1:-1]

            # Expected Value
            w_avg = current_price
            implied_move = 0
            if np.sum(pdf) > 0:
                w_avg = np.average(pdf_strikes, weights=pdf)
                implied_move = np.sqrt(np.average((pdf_strikes - w_avg)**2, weights=pdf))
            
            # Greeks
            atm_iv = float(iv_spline(current_price))
            greeks = black_scholes_greeks(current_price, current_price, T, RISK_FREE_RATE, atm_iv)

            all_pdfs.append(pd.DataFrame({
                'Strike': pdf_strikes, 'Probability': pdf, 'Expiration': exp_str,
                'ExpectedPrice': w_avg, 'ImpliedMove': implied_move,
                'Delta': greeks['delta'], 'Gamma': greeks['gamma'], 'Theta': greeks['theta']
            }))
        except: continue
    
    progress_bar.empty()
    
    if not all_pdfs: return None, None, None

    df = pd.concat(all_pdfs)
    avg_iv = np.mean(all_ivs) if all_ivs else 0
    pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0
    
    meta = {
        "earnings": earnings_date,
        "rank": vol_rank,
        "pcr": pcr,
        "avg_iv": avg_iv
    }
    return df, current_price, meta

def render_heatmap(df, current_price, meta, ticker, mode, ai_mode):
    # Binning
    step = 10 if mode == "Long" else (5 if current_price > 200 else 1)
    bins = np.arange(np.floor(df['Strike'].min()), np.ceil(df['Strike'].max()), step)
    df['StrikeBin'] = pd.cut(df['Strike'], bins=bins, labels=bins[:-1])
    
    heatmap_data = df.groupby(['Expiration', 'StrikeBin'])['Probability'].mean().reset_index()
    matrix = heatmap_data.pivot(index='StrikeBin', columns='Expiration', values='Probability')
    matrix.columns = pd.to_datetime(matrix.columns)
    matrix = matrix.sort_index(axis=1)
    matrix.columns = matrix.columns.strftime('%Y-%m-%d')
    matrix = matrix.sort_index(ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    # Huge top margin for HUD
    plt.subplots_adjust(top=0.80, bottom=0.15)

    # Zebra Striping
    for i in range(0, len(matrix.index), 2):
        ax.axhspan(i, i+1, color='white', alpha=0.05, zorder=2)
        ax.axhspan(i+1, i+2, color='black', alpha=0.05, zorder=2)

    kws = {'linewidths': 0.05, 'linecolor': 'black'} if ai_mode else {}
    sns.heatmap(matrix, cmap="magma", cbar_kws={'label': 'Probability Density'}, ax=ax, **kws)

    # Trendline
    price_map = df.groupby('Expiration')['ExpectedPrice'].first()
    trend_prices = []
    x_coords = []
    for i, col in enumerate(matrix.columns):
        if col in price_map:
            trend_prices.append(price_map[col])
            x_coords.append(i + 0.5)
            
    trend_color = 'cyan'
    trend_label = "NEUTRAL"
    if len(trend_prices) > 1:
        if trend_prices[-1] > trend_prices[0]:
            trend_color = '#00ff00'; trend_label = "BULLISH"
        else:
            trend_color = '#ff3333'; trend_label = "BEARISH"
            
    strikes_y = np.array([float(x) for x in matrix.index])
    y_coords = []
    if len(strikes_y) > 1:
        strikes_asc = strikes_y[::-1]
        indices_desc = np.arange(len(strikes_y))[::-1]
        for p in trend_prices:
            y_coords.append(np.interp(p, strikes_asc, indices_desc) + 0.5)
            
        plt.plot(x_coords, y_coords, color=trend_color, marker='o', linestyle='-', linewidth=2.5, label=f'Trend: {trend_label}')
        if trend_prices:
             plt.text(x_coords[-1]+0.2, y_coords[-1], f"${trend_prices[-1]:.2f}", color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, edgecolor=trend_color))

    # Annotations
    move_map = df.groupby('Expiration')['ImpliedMove'].first()
    delta_map = df.groupby('Expiration')['Delta'].first()
    gamma_map = df.groupby('Expiration')['Gamma'].first()
    theta_map = df.groupby('Expiration')['Theta'].first()

    for i, col in enumerate(matrix.columns):
        if col in move_map:
            # Expected Move
            ax.text(i+0.5, -0.2, f"±${move_map[col]:.1f}", ha='center', va='bottom', color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.5, pad=0))
            # Greeks
            ax.text(i+0.5, -1.5, f"{{Δ{delta_map[col]:.2f} Γ{gamma_map[col]:.2f} Θ{theta_map[col]:.2f}}}", ha='left', va='bottom', color='#AAAAAA', fontsize=8, rotation=45, fontweight='bold')

    # Hotspots
    for c in range(matrix.shape[1]):
        col_data = matrix.iloc[:, c]
        r = col_data.argmax()
        val = col_data.iloc[r] * step
        if val > 0.01:
            ax.text(c+0.5, r+0.5, f"{val:.0%}", ha='center', va='center', color='black', fontsize=8, fontweight='bold', bbox=dict(boxstyle="circle,pad=0.1", fc="white", alpha=0.6))

    # Spot Line
    try:
        y_pos = list(matrix.index).index(min(strikes_y, key=lambda x: abs(x-current_price)))
        ax.axhline(y_pos+0.5, color='cyan', linestyle='--', linewidth=1.5, label='Spot')
    except: pass

    # HUD on Chart
    plt.text(0, 1.25, f"  {ticker} ANALYSIS HUD  ", transform=ax.transAxes, fontsize=14, fontweight='bold', color='white', backgroundcolor='#333333')
    plt.text(0, 1.20, f"Price: ${current_price:.2f}  |  Trend: {trend_label}  |  PCR: {meta['pcr']:.2f}", transform=ax.transAxes, fontsize=11, fontweight='bold')
    plt.text(0, 1.16, f"Avg IV: {meta['avg_iv']*100:.1f}%  |  IV Rank: {meta['rank']}  |  Earn: {meta['earnings']}", transform=ax.transAxes, fontsize=11)

    # Footer
    if ai_mode:
        plt.figtext(0.5, 0.02, "AI NOTE: Cyan=Spot. Green/Red=Trend. {} = ATM Greeks. Hotspots=High Prob. ±$ = 1 Std Dev.", ha="center", fontsize=10, bbox={"facecolor":"cyan", "alpha":0.1})

    # Colorbar Marker
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.plot([0,1], [0.5,0.5], color='white', transform=cbar.ax.transAxes)
        cbar.ax.text(1.1, 0.5, '50%', transform=cbar.ax.transAxes, va='center', fontsize=8)

    plt.title(f"Market Implied Probability: {ticker} ({mode})", fontsize=16, y=1.28)
    plt.xlabel("Expiration Date"); plt.ylabel("Strike Price")
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    
    # Bold Y-Axis
    for l in ax.get_yticklabels():
        try: 
            if float(l.get_text()) % 10 == 0: l.set_fontweight('bold'); l.set_fontsize(11)
        except: pass

    return fig

# ==========================================
#  UI LAYOUT
# ==========================================

# Sidebar Controls
st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="CVX").upper()
mode = st.sidebar.selectbox("Expiration Mode", ["Short", "Medium", "Long", "All"], index=0)
smoothing = st.sidebar.slider("Smoothing Factor", 0.1, 2.0, 0.5, 0.1)
ai_mode = st.sidebar.checkbox("AI Readability Mode", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Press 'R' to rerun if the chart doesn't update automatically.")

# Main Dashboard
st.title(f"OPTION PORTAL: {ticker}")

if ticker:
    with st.spinner('Fetching Market Data & Calculating Greeks...'):
        df, current_price, meta = get_market_data(ticker, mode, smoothing)

    if df is not None:
        # Top Level Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Spot Price", f"${current_price:.2f}")
        col2.metric("Put/Call Ratio", f"{meta['pcr']:.2f}")
        col3.metric("Avg IV", f"{meta['avg_iv']*100:.1f}%")
        col4.metric("IV Rank", meta['rank'])

        # Chart
        fig = render_heatmap(df, current_price, meta, ticker, mode, ai_mode)
        st.pyplot(fig)
        
        # Data Table
        with st.expander("Show Raw Data"):
            st.dataframe(df)
            
    else:
        st.error("Could not fetch data. Ticker might be invalid or no options available.")