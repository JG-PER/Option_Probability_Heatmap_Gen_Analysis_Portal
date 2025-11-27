# **Option Probability Heatmap Generator & Analysis Portal**

## **Overview**

This project provides a suite of tools to visualize **Market Implied Probability Distributions** derived from option chains. By calculating the second derivative of option prices (Breeden-Litzenberger formula) and smoothing Implied Volatility (IV) smiles, these tools generate heatmaps that show where the market expects a stock price to go.

The system includes standalone generation scripts, a single-file interactive dashboard (Streamlit), and a decoupled web portal architecture (Flask \+ HTML).

## **File Structure & Descriptions**

### **1\. Core Logic & Scripts**

* **core\_chart\_gen.py**: The standalone Python script that fetches data, performs the Black-Scholes calculations, and renders a static heatmap using Matplotlib/Seaborn. This is the foundation for all other builds.  
* **streamlit\_dashboard\_build.py**: A complete, single-file interactive web application. It wraps the core logic in a **Streamlit** interface, allowing for real-time toggling of tickers, timeframes, and visual settings without needing to edit code.  
* **flask\_portal\_build.py**: A backend API server using **Flask**. This decouples the logic from the view, serving JSON data and base64-encoded images to a separate frontend (e.g., portal.html).

### **2\. Automation & Launchers**

* **streamlit\_launcher.sh**: A Bash script that automates the dependency check and execution of the Streamlit dashboard. Run this to launch the interactive UI immediately.  
* **flask\_launcher.sh**: A Bash script that installs flask-cors and launches the Flask API backend. It prepares the local server to receive requests from the HTML portal.

## **Installation & Setup**

All tools require Python 3.8+ and the following dependencies:

pip install yfinance pandas numpy scipy matplotlib seaborn streamlit flask flask-cors

### **Quick Start**

**To run the Interactive Dashboard:**

./streamlit\_launcher.sh

**To run the HTML Portal Backend:**

./flask\_launcher.sh  
\# Then open 'portal.html' in your browser.

## **AI Agent Integration Prompt**

Add the following paragraph to your AI agent's system prompt or context window when asking it to analyze these charts. This ensures the AI understands exactly how the visual data is constructed and how to interpret the specific annotations used in this project.

### **Core Context Prompt for AI Agents:**

"You are analyzing a Market Implied Probability Heatmap constructed using the Breeden-Litzenberger method, where brighter or hotter regions on the chart represent the 'Probability Mass' indicating price zones with the highest market-priced likelihood of expiration. The chart features a Cyan Line representing the current Spot Price and a Green (Bullish) or Red (Bearish) Trendline that connects the probability-weighted 'Expected Price' for each expiration, visualizing the market consensus 'Center of Gravity' over time. The top of each expiration column displays the Expected Move (±1 Standard Deviation) and the At-The-Money (ATM) Greeks formatted as {Delta, Gamma, Theta}, while the Head-Up Display (HUD) provides the IV Rank to gauge relative option expensiveness and the Put/Call Ratio to assess sentiment flow. When requesting analysis or chart generation, you may specify the Ticker for any US equity, the Expiration Mode as 'Short' for tactical views under 45 days, 'Medium' for swing views up to 6 months, or 'Long' for investment views up to 2 years, and a Smoothing factor between 0.1 and 2.0 to adjust the spline fit on the Volatility Smile from raw/noisy to idealized."

## **Technical Methodology**

1. **Data Ingestion:** Fetches real-time Option Chains via yfinance.  
2. **Volatility Smoothing:** Fits a Cubic Spline to the Implied Volatility (IV) vs. Strike curve to remove market noise and bid-ask spread artifacts.  
3. **Price Reconstruction:** Converts smoothed IV back into theoretical Call Prices using Black-Scholes.  
4. **Probability Extraction:** Calculates the Probability Density Function (PDF) using the second finite difference of the Call Prices.  
5. **Greek Calculation:** Computes Delta, Gamma, and Theta for the At-The-Money (ATM) strike for every expiration date to assess risk exposure.