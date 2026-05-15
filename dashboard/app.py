"""
dashboard/app.py — Streamlit Real-Time Risk Dashboard.

Features:
- Connects to FastAPI WebSocket for real-time predictions
- Calculates Portfolio VaR via REST API
- Displays interactive Plotly charts and heatmaps
"""
import json
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import websocket

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Real-Time Portfolio Risk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2A2A3C;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00E676;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .warning-value { color: #FF3D00; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

WS_URL = "ws://localhost:8000/ws/risk-stream"
API_URL = "http://localhost:8000/api"

# --- STATE INITIALISATION ---
if "history" not in st.session_state:
    st.session_state.history = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "ws" not in st.session_state:
    try:
        st.session_state.ws = websocket.create_connection(WS_URL)
        st.session_state.ws.settimeout(0.5)
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        st.session_state.ws = None

# --- SIDEBAR: PORTFOLIO WEIGHTS ---
st.sidebar.title("Portfolio Configuration")
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

st.sidebar.subheader("Asset Weights")
weights = {}
total_weight = 0
for sym in symbols:
    w = st.sidebar.slider(f"{sym} Weight (%)", 0, 100, 20)
    weights[sym] = w / 100.0
    total_weight += w

if total_weight != 100:
    st.sidebar.warning(f"Total weight is {total_weight}%. Please adjust to 100%.")

# --- REAL-TIME DATA FETCHING ---
def fetch_realtime_data():
    if st.session_state.ws:
        try:
            msg = st.session_state.ws.recv()
            data = json.loads(msg)
            
            # Store in history
            st.session_state.history.append(data)
            # Keep last 100 ticks
            if len(st.session_state.history) > 100:
                st.session_state.history.pop(0)
                
            st.session_state.metrics = data.get("predictions", {})
        except websocket.WebSocketTimeoutException:
            pass
        except Exception as e:
            st.warning("WebSocket connection lost. Reconnecting...")
            try:
                st.session_state.ws = websocket.create_connection(WS_URL)
                st.session_state.ws.settimeout(0.5)
            except:
                pass

fetch_realtime_data()

# --- MAIN DASHBOARD ---
st.title("⚡ Real-Time Portfolio Risk Engine")
st.markdown("Powered by Temporal Fusion Transformer & Apache Kafka")

if not st.session_state.metrics:
    st.info("⏳ Waiting for model warm-up and real-time predictions...")
    time.sleep(1)
    st.rerun()

# 1. Fetch Portfolio Level Risk
try:
    resp = requests.post(f"{API_URL}/portfolio", json={"weights": weights})
    if resp.status_code == 200:
        port_risk = resp.json()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio VaR (99%)</div>
                <div class="metric-value warning-value">{(port_risk['portfolio_var_99'] * 100):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio VaR (95%)</div>
                <div class="metric-value warning-value">{(port_risk['portfolio_var_95'] * 100):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Volatility</div>
                <div class="metric-value">{(port_risk['portfolio_vol_forecast'] * 100):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not calculate portfolio risk: {e}")

# 2. Per-Asset Metrics Table & Chart
st.subheader("Asset-Level Risk Metrics")
preds = st.session_state.metrics

df_metrics = pd.DataFrame([
    {
        "Symbol": sym, 
        "VaR 99%": f"{(v['var_99']*100):.2f}%",
        "VaR 95%": f"{(v['var_95']*100):.2f}%", 
        "Volatility": f"{(v['vol_forecast']*100):.2f}%"
    }
    for sym, v in preds.items()
])

colA, colB = st.columns([1, 2])
with colA:
    st.dataframe(df_metrics, hide_index=True, width='stretch')

with colB:
    # Bar chart of VaR 99%
    plot_df = pd.DataFrame([{"Symbol": k, "VaR 99%": v["var_99"]*100} for k, v in preds.items()])
    fig = px.bar(
        plot_df, x="Symbol", y="VaR 99%", 
        title="Value at Risk (99%) by Asset",
        color="VaR 99%", color_continuous_scale="Reds",
        template="plotly_dark"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# 3. Dynamic Time-Series Chart
if len(st.session_state.history) > 5:
    st.subheader("Real-Time VaR Trajectory")
    
    # Flatten history
    hist_data = []
    for item in st.session_state.history:
        ts = item["timestamp"]
        for sym, metrics in item["predictions"].items():
            hist_data.append({
                "Time": ts,
                "Symbol": sym,
                "VaR 99": metrics["var_99"] * 100
            })
            
    df_hist = pd.DataFrame(hist_data)
    if not df_hist.empty:
        fig_line = px.line(
            df_hist, x="Time", y="VaR 99", color="Symbol",
            title="VaR 99% Over Time",
            template="plotly_dark"
        )
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)

# Auto-refresh loop to poll WebSocket frequently
time.sleep(1)
st.rerun()
