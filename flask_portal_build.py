# ------------------------------------------------------------------
# OPTION ANALYSIS PORTAL
# A Flask-based web interface for the Option Probability Heatmap
# ------------------------------------------------------------------
# DEPENDENCIES:
# pip install flask yfinance pandas numpy scipy matplotlib seaborn
# ------------------------------------------------------------------

from flask import Flask, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ==========================================
#  CORE ANALYTICS LOGIC (From option_heatmap.py)
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

def get_earnings_date(ticker_obj):
    try:
        earnings = ticker_obj.earnings_dates
        if earnings is not None and not earnings.empty:
            future = earnings[earnings.index > datetime.now()]
            if not future.empty: return future.index[-1].strftime('%b %d')
    except: pass
    return "N/A"

def get_volatility_stats(ticker_obj, current_iv):
    try:
        hist = ticker_obj.history(period="1y")
        if hist.empty: return "N/A"
        hist['LogRet'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['Vol'] = hist['LogRet'].rolling(window=30).std() * np.sqrt(252)
        min_v, max_v = hist['Vol'].min(), hist['Vol'].max()
        if max_v != min_v:
            return f"{((current_iv - min_v) / (max_v - min_v) * 100):.0f}%"
    except: pass
    return "N/A"

def generate_heatmap_data(ticker_symbol, mode, smoothing):
    ticker = yf.Ticker(ticker_symbol)
    try:
        hist = ticker.history(period="1d")
        if hist.empty: return None
        current_price = hist['Close'].iloc[-1]
    except: return None

    # Filter Expirations
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

    if not target_exps: return None

    all_pdfs = []
    all_ivs = []
    total_call_vol = 0
    total_put_vol = 0

    for exp_str in target_exps:
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
            
            # Smoothing & PDF
            calls = calls.sort_values('strike')
            strikes = calls['strike'].values
            ivs = calls['impliedVolatility'].values
            iv_spline = UnivariateSpline(strikes, ivs, k=3, s=float(smoothing))
            
            # Days to Exp
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - today).days / 365.0
            if T <= 0: continue

            dense_strikes = np.linspace(strikes.min(), strikes.max(), 200)
            smoothed_ivs = iv_spline(dense_strikes)
            prices = np.array([black_scholes_call_price(current_price, k, T, RISK_FREE_RATE, iv) for k, iv in zip(dense_strikes, smoothed_ivs)])
            
            dk = dense_strikes[1] - dense_strikes[0]
            pdf = np.maximum(np.diff(prices, 2) / (dk**2) * np.exp(RISK_FREE_RATE * T), 0)
            pdf_strikes = dense_strikes[1:-1]

            # Stats
            w_avg = current_price
            implied_move = 0
            if np.sum(pdf) > 0:
                w_avg = np.average(pdf_strikes, weights=pdf)
                implied_move = np.sqrt(np.average((pdf_strikes - w_avg)**2, weights=pdf))
            
            atm_iv = float(iv_spline(current_price))
            greeks = black_scholes_greeks(current_price, current_price, T, RISK_FREE_RATE, atm_iv)

            all_pdfs.append(pd.DataFrame({
                'Strike': pdf_strikes, 'Probability': pdf, 'Expiration': exp_str,
                'ExpectedPrice': w_avg, 'ImpliedMove': implied_move,
                'Delta': greeks['delta'], 'Gamma': greeks['gamma'], 'Theta': greeks['theta']
            }))
        except: continue

    if not all_pdfs: return None
    
    df = pd.concat(all_pdfs)
    avg_iv = np.mean(all_ivs) if all_ivs else 0
    pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0
    meta = {
        "earnings": get_earnings_date(ticker),
        "rank": get_volatility_stats(ticker, avg_iv),
        "pcr": pcr,
        "avg_iv": avg_iv
    }
    return df, current_price, meta

def create_plot(df, current_price, meta, ticker, mode, ai_mode):
    # Setup Data
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
    plt.subplots_adjust(top=0.82, bottom=0.15)

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

    # Annotations (Greeks / Move / Hotspots)
    move_map = df.groupby('Expiration')['ImpliedMove'].first()
    delta_map = df.groupby('Expiration')['Delta'].first()
    gamma_map = df.groupby('Expiration')['Gamma'].first()
    theta_map = df.groupby('Expiration')['Theta'].first()

    for i, col in enumerate(matrix.columns):
        if col in move_map:
            ax.text(i+0.5, -0.2, f"±${move_map[col]:.1f}", ha='center', va='bottom', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, pad=0))
            ax.text(i+0.5, -1.5, f"{{Δ{delta_map[col]:.2f} Γ{gamma_map[col]:.2f} Θ{theta_map[col]:.2f}}}", ha='left', va='bottom', color='#AAAAAA', fontsize=7, rotation=45, fontweight='bold')

    for c in range(matrix.shape[1]):
        col_data = matrix.iloc[:, c]
        r = col_data.argmax()
        val = col_data.iloc[r] * step
        if val > 0.01:
            ax.text(c+0.5, r+0.5, f"{val:.0%}", ha='center', va='center', color='black', fontsize=7, fontweight='bold', bbox=dict(boxstyle="circle,pad=0.1", fc="white", alpha=0.6))

    # Spot Line
    try:
        y_pos = list(matrix.index).index(min(strikes_y, key=lambda x: abs(x-current_price)))
        ax.axhline(y_pos+0.5, color='cyan', linestyle='--', linewidth=1.5, label='Spot')
    except: pass

    # HUD
    plt.text(0, 1.25, f"  {ticker} ANALYSIS HUD  ", transform=ax.transAxes, fontsize=14, fontweight='bold', color='white', backgroundcolor='#333333')
    plt.text(0, 1.20, f"Price: ${current_price:.2f}  |  Trend: {trend_label}  |  PCR: {meta['pcr']:.2f}", transform=ax.transAxes, fontsize=11, fontweight='bold')
    plt.text(0, 1.16, f"Avg IV: {meta['avg_iv']*100:.1f}%  |  IV Rank: {meta['rank']}  |  Earn: {meta['earnings']}", transform=ax.transAxes, fontsize=11)

    # Footer
    if ai_mode:
        plt.figtext(0.5, 0.02, "AI NOTE: Cyan=Spot. Green/Red=Trend. {} = ATM Greeks. Hotspots=High Prob.", ha="center", fontsize=10, bbox={"facecolor":"cyan", "alpha":0.1})

    # Colorbar 50% mark
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.plot([0,1], [0.5,0.5], color='white', transform=cbar.ax.transAxes)
        cbar.ax.text(1.1, 0.5, '50%', transform=cbar.ax.transAxes, va='center', fontsize=8)

    # Formatting
    plt.title(f"Market Implied Probability: {ticker} ({mode})", fontsize=16, y=1.28)
    plt.xlabel("Expiration"); plt.ylabel("Strike")
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    
    # Bold Y-Axis
    for l in ax.get_yticklabels():
        try: 
            if float(l.get_text()) % 10 == 0: l.set_fontweight('bold'); l.set_fontsize(11)
        except: pass

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ==========================================
#  WEB INTERFACE
# ==========================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Option Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f1115; color: #e2e8f0; font-family: 'Courier New', monospace; }
        .input-dark { background-color: #1e293b; border: 1px solid #334155; color: #fff; }
        .btn-glow:hover { box-shadow: 0 0 15px #3b82f6; }
        .loading { display: none; }
    </style>
</head>
<body class="min-h-screen p-6">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="flex justify-between items-center mb-8 border-b border-gray-700 pb-4">
            <div>
                <h1 class="text-3xl font-bold text-blue-400">OPTION <span class="text-white">PORTAL</span></h1>
                <p class="text-xs text-gray-500 mt-1">MARKET IMPLIED PROBABILITY & GREEKS ENGINE</p>
            </div>
            <div class="text-right">
                <div class="text-green-400 text-sm">● SYSTEM READY</div>
                <div class="text-xs text-gray-600">v2.4.0 AI_BUILD</div>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <label class="block text-xs text-blue-300 mb-1">TICKER SYMBOL</label>
                <input type="text" id="ticker" value="CVX" class="w-full input-dark rounded p-2 font-bold text-lg uppercase focus:outline-none focus:border-blue-500">
            </div>

            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <label class="block text-xs text-blue-300 mb-1">EXPIRATION MODE</label>
                <select id="mode" class="w-full input-dark rounded p-2 focus:outline-none">
                    <option value="Short">SHORT (Next 45 Days)</option>
                    <option value="Medium">MEDIUM (1-6 Months)</option>
                    <option value="Long">LONG (1-2 Years)</option>
                    <option value="All">ALL EXPIRATIONS</option>
                </select>
            </div>

            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg">
                <label class="block text-xs text-blue-300 mb-1">SMOOTHING (0.1 - 2.0)</label>
                <input type="number" id="smoothing" value="0.5" step="0.1" class="w-full input-dark rounded p-2 focus:outline-none">
            </div>

            <div class="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg flex items-center justify-between">
                <div>
                    <label class="block text-xs text-blue-300 mb-1">AI READABILITY</label>
                    <p class="text-[10px] text-gray-500">Enhance Visuals</p>
                </div>
                <input type="checkbox" id="aiMode" checked class="w-6 h-6 text-blue-600 rounded focus:ring-blue-500 bg-gray-700 border-gray-600">
            </div>
        </div>

        <!-- Action Bar -->
        <div class="flex justify-center mb-8">
            <button onclick="generate()" class="btn-glow bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-12 rounded transition-all duration-300 flex items-center">
                <span id="btnText">INITIATE ANALYSIS SEQUENCE</span>
                <svg id="loadingIcon" class="loading animate-spin ml-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            </button>
        </div>

        <!-- Output Display -->
        <div class="bg-black rounded-xl border border-gray-800 p-2 min-h-[600px] flex items-center justify-center relative overflow-hidden">
            <div id="placeholder" class="text-center text-gray-600">
                <p class="text-4xl mb-2">WAITING FOR INPUT</p>
                <p class="text-sm">Select parameters above and initialize.</p>
            </div>
            <img id="resultImage" class="hidden w-full h-auto rounded shadow-2xl" src="" alt="Analysis Result">
        </div>
        
        <div id="statusLog" class="mt-4 font-mono text-xs text-green-500"></div>
    </div>

    <script>
        async function generate() {
            const btn = document.querySelector('button');
            const btnText = document.getElementById('btnText');
            const loader = document.getElementById('loadingIcon');
            const img = document.getElementById('resultImage');
            const placeholder = document.getElementById('placeholder');
            const status = document.getElementById('statusLog');

            // UI Loading State
            btn.disabled = true;
            btn.classList.add('opacity-50');
            btnText.innerText = "PROCESSING MARKET DATA...";
            loader.style.display = 'block';
            status.innerText = "> Connecting to Exchange Data API...";

            const payload = {
                ticker: document.getElementById('ticker').value.toUpperCase(),
                mode: document.getElementById('mode').value,
                smoothing: document.getElementById('smoothing').value,
                aiMode: document.getElementById('aiMode').checked
            };

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (data.error) {
                    status.innerText = "> ERROR: " + data.error;
                    alert("Error: " + data.error);
                } else {
                    status.innerText = "> Data Received. Rendering Visualization...";
                    img.src = "data:image/png;base64," + data.image;
                    img.classList.remove('hidden');
                    placeholder.classList.add('hidden');
                    status.innerText = "> ANALYSIS COMPLETE: " + payload.ticker;
                }
            } catch (e) {
                status.innerText = "> SYSTEM FAILURE: Check connection.";
                console.error(e);
            } finally {
                // Reset UI
                btn.disabled = false;
                btn.classList.remove('opacity-50');
                btnText.innerText = "INITIATE ANALYSIS SEQUENCE";
                loader.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    ticker = data.get('ticker', 'SPY')
    mode = data.get('mode', 'Short')
    smoothing = data.get('smoothing', 0.5)
    ai_mode = data.get('aiMode', True)

    try:
        df, price, meta = generate_heatmap_data(ticker, mode, smoothing)
        if df is None:
            return jsonify({'error': 'No data found or invalid ticker'})
        
        img_str = create_plot(df, price, meta, ticker, mode, ai_mode)
        return jsonify({'image': img_str})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("STARTING OPTION PORTAL...")
    print("OPEN YOUR BROWSER TO: http://127.0.0.1:5000")
    app.run(debug=True)