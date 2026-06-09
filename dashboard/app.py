"""
dashboard/app.py — Streamlit Real-Time Risk Dashboard.

Features:
- Connects to FastAPI WebSocket for real-time predictions
- Calculates Portfolio VaR via REST API (covariance-based, not weighted average)
- Displays parametric VaR, historical VaR, CVaR, and correlation/covariance heatmaps
- Shows per-asset risk metrics from TFT model
"""
import json
import time
import requests
import pandas as pd
import numpy as np
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

from dashboard.auth import check_authentication, is_admin, logout
check_authentication()

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
    .reliable-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .reliable-yes { background: #1B5E20; color: #A5D6A7; }
    .reliable-no  { background: #4A1010; color: #EF9A9A; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

WS_URL  = "ws://localhost:8000/ws/risk-stream"
API_URL = "http://localhost:8000/api"

# --- STATE INITIALISATION ---
if "history" not in st.session_state:
    st.session_state.history = []          # List[dict] mỗi item là 1 WebSocket payload
if "history_seen_keys" not in st.session_state:
    st.session_state.history_seen_keys = set()  # Tập hợp key (timestamp, symbol) đã thấy
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "ws" not in st.session_state:
    try:
        st.session_state.ws = websocket.create_connection(WS_URL)
        st.session_state.ws.settimeout(0.5)
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        st.session_state.ws = None


# ─────────────────────────────────────────────────────────────
# HELPER: Chuẩn bị dữ liệu sạch cho biểu đồ VaR Trajectory
# ─────────────────────────────────────────────────────────────
MAX_POINTS_PER_SYMBOL = 200  # Giới hạn số điểm mỗi mã cổ phiếu


def prepare_var_trajectory_data(history: list, metric_col: str) -> pd.DataFrame:
    """Chuyển đổi raw history thành DataFrame sạch để vẽ biểu đồ.

    Args:
        history: List các WebSocket payload dict.
        metric_col: Tên cột cần lấy từ predictions, ví dụ 'var_99', 'var_95', 'vol_forecast'.

    Returns:
        DataFrame với cột: [timestamp, symbol, value_pct] đã parse, sort, dedup.
        Trả về DataFrame rỗng nếu không đủ dữ liệu.
    """
    REQUIRED_PRED_KEYS = {metric_col}

    rows = []
    for item in history:
        ts_raw = item.get("timestamp")
        preds = item.get("predictions", {})
        if not ts_raw or not preds:
            continue
        for sym, m in preds.items():
            if not REQUIRED_PRED_KEYS.issubset(m.keys()):
                continue
            rows.append({
                "timestamp": ts_raw,
                "symbol": sym,
                "value": m[metric_col],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 1. Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return df

    # 2. Dedup theo (timestamp, symbol) — giữ lại hàng cuối cùng
    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")

    # 3. Sort theo symbol rồi timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # 4. Giới hạn MAX_POINTS_PER_SYMBOL điểm gần nhất mỗi mã
    df = df.groupby("symbol", group_keys=False).tail(MAX_POINTS_PER_SYMBOL)

    # 5. Chuyển sang phần trăm để hiển thị (nhân 100 đúng một lần)
    df["value_pct"] = df["value"] * 100

    return df

# --- SIDEBAR: PORTFOLIO WEIGHTS ---
st.sidebar.title("Portfolio Configuration")

st.sidebar.markdown(f"👤 **Xin chào, {st.session_state.get('username')}**")
st.sidebar.markdown(f"**Quyền:** {'Admin' if is_admin() else 'User'}")
if st.sidebar.button("🚪 Đăng xuất", type="secondary"):
    logout()

# --- ADMIN ROUTING ---
app_mode = "User Dashboard"
if is_admin():
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Chế độ Xem", ["Admin Dashboard", "User Dashboard"])

if app_mode == "Admin Dashboard":
    from dashboard.admin import show_admin_dashboard
    show_admin_dashboard()
    st.stop()  # Stop rendering the User Dashboard

from config.settings import get_settings
cfg = get_settings()

st.sidebar.markdown("---")
symbols = cfg.tickers_list

st.sidebar.subheader("Asset Weights (%)")
weights = {}
total_weight = 0
for sym in symbols:
    w = st.sidebar.slider(f"{sym}", 0, 100, 20)
    weights[sym] = w / 100.0
    total_weight += w

weight_ok = total_weight == 100
if not weight_ok:
    st.sidebar.warning(f"Total weight: {total_weight}% (phải bằng 100%)")

st.sidebar.markdown("---")
cov_window = st.sidebar.slider(
    "Covariance Window (bars)",
    min_value=30,
    max_value=500,
    value=390,
    help="Số bar 1-phút dùng để tính covariance matrix. Khuyến nghị ≥ 390 để ổn định.",
)

# --- REAL-TIME DATA FETCHING ---
def fetch_realtime_data():
    """Nhận 1 message từ WebSocket và append vào history, có chống trùng lặp."""
    if not st.session_state.ws:
        return
    try:
        msg = st.session_state.ws.recv()
        data = json.loads(msg)

        ts = data.get("timestamp", "")
        preds = data.get("predictions", {})

        # Kiểm tra xem payload này có điểm dữ liệu mới không
        new_keys = {
            (ts, sym) for sym in preds
            if (ts, sym) not in st.session_state.history_seen_keys
        }

        if new_keys:
            st.session_state.history.append(data)
            st.session_state.history_seen_keys.update(new_keys)

            # Giữ tối đa 500 payload gần nhất (MAX_POINTS_PER_SYMBOL * N_symbols)
            if len(st.session_state.history) > 500:
                # Xoá payload cũ và đồng bộ lại seen_keys
                removed = st.session_state.history.pop(0)
                removed_ts = removed.get("timestamp", "")
                for sym in removed.get("predictions", {}):
                    st.session_state.history_seen_keys.discard((removed_ts, sym))

        st.session_state.metrics = preds

    except websocket.WebSocketTimeoutException:
        pass
    except Exception:
        st.warning("WebSocket connection lost. Reconnecting...")
        try:
            st.session_state.ws = websocket.create_connection(WS_URL)
            st.session_state.ws.settimeout(0.5)
        except Exception:
            pass


fetch_realtime_data()

# --- MAIN DASHBOARD ---
st.title("⚡ Real-Time Portfolio Risk Engine")
st.markdown(
    "Powered by **Temporal Fusion Transformer** & **Apache Kafka** · "
    "Portfolio risk computed via covariance matrix (parametric + historical VaR)"
)

if not st.session_state.get("metrics"):
    st.info("⏳ Waiting for model warm-up and real-time predictions...")
    time.sleep(1)
    st.rerun()

# ─────────────────────────────────────────────────────────────
# 1. PORTFOLIO LEVEL RISK (covariance-based)
# ─────────────────────────────────────────────────────────────
st.subheader("📊 Portfolio Risk Metrics")

try:
    resp = requests.post(
        f"{API_URL}/portfolio",
        json={"weights": weights, "covariance_window": cov_window},
        timeout=5,
    )
    if resp.status_code == 200:
        port_risk = resp.json()

        # Warning / reliability banner
        if port_risk.get("warning"):
            st.warning(f"⚠️ {port_risk['warning']}")

        reliable = port_risk.get("reliable", False)
        method   = port_risk.get("method_used", "unknown")
        n_obs    = port_risk.get("n_observations", 0)
        badge_cls  = "reliable-yes" if reliable else "reliable-no"
        badge_text = "✅ Reliable" if reliable else "⚠️ Provisional"
        st.markdown(
            f'<span class="reliable-badge {badge_cls}">{badge_text}</span>'
            f' &nbsp; Method: <code>{method}</code> &nbsp; Observations: <code>{n_obs}</code>',
            unsafe_allow_html=True,
        )

        # ── Main metrics row ────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Parametric VaR (99%)</div>
                <div class="metric-value warning-value">{port_risk['portfolio_var_99_parametric']*100:.3f}%</div>
                <div class="metric-label">Normal distribution</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Parametric VaR (95%)</div>
                <div class="metric-value warning-value">{port_risk['portfolio_var_95_parametric']*100:.3f}%</div>
                <div class="metric-label">Normal distribution</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Horizon Volatility</div>
                <div class="metric-value">{port_risk['portfolio_volatility_horizon']*100:.4f}%</div>
                <div class="metric-label">Per 1-minute bar</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Annualized Volatility</div>
                <div class="metric-value">{port_risk['portfolio_volatility_annualized']*100:.2f}%</div>
                <div class="metric-label">×√(252×390)</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Historical VaR + CVaR row ───────────────────────────
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Historical VaR (99%)</div>
                <div class="metric-value warning-value">{port_risk['portfolio_var_99_historical']*100:.3f}%</div>
                <div class="metric-label">Empirical quantile</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Historical VaR (95%)</div>
                <div class="metric-value warning-value">{port_risk['portfolio_var_95_historical']*100:.3f}%</div>
                <div class="metric-label">Empirical quantile</div>
            </div>
            """, unsafe_allow_html=True)
        with c7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CVaR / ES (95%)</div>
                <div class="metric-value warning-value">{port_risk['portfolio_cvar_95']*100:.3f}%</div>
                <div class="metric-label">Expected Shortfall</div>
            </div>
            """, unsafe_allow_html=True)
        with c8:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CVaR / ES (99%)</div>
                <div class="metric-value warning-value">{port_risk['portfolio_cvar_99']*100:.3f}%</div>
                <div class="metric-label">Expected Shortfall</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Correlation Heatmap ─────────────────────────────────
        corr_data = port_risk.get("correlation_matrix", {})
        if corr_data:
            st.subheader("🔗 Correlation Matrix")
            corr_df = pd.DataFrame(corr_data)
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns.tolist(),
                y=corr_df.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                zmin=-1, zmax=1,
                text=corr_df.round(3).values,
                texttemplate="%{text}",
                colorbar=dict(title="Correlation"),
            ))
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10),
                height=350,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.error(f"Portfolio API error {resp.status_code}: {resp.text[:200]}")

except requests.exceptions.ConnectionError:
    st.error("Cannot connect to FastAPI backend (localhost:8000). Is the backend running?")
except Exception as e:
    st.error(f"Could not calculate portfolio risk: {e}")

# ─────────────────────────────────────────────────────────────
# 2. PER-ASSET METRICS TABLE & CHART
# ─────────────────────────────────────────────────────────────
st.subheader("📋 Asset-Level Risk Metrics (TFT Model)")
preds = st.session_state.get("metrics", {})

if preds:
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
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    with colB:
        plot_df = pd.DataFrame([
            {"Symbol": k, "VaR 99%": v["var_99"] * 100} for k, v in preds.items()
        ])
        fig = px.bar(
            plot_df, x="Symbol", y="VaR 99%",
            title="Per-Asset VaR 99% (TFT model output)",
            color="VaR 99%", color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chờ dữ liệu từ model...")

# ─────────────────────────────────────────────────────────────
# 3. REAL-TIME VaR TRAJECTORY (Per-Asset) — CLEANED
# ─────────────────────────────────────────────────────────────
st.subheader("📈 Real-Time VaR Trajectory (Per-Asset)")
st.caption(
    "Biểu đồ này thể hiện ước tính rủi ro từng mã cổ phiếu riêng lẻ (asset-level VaR). "
    "Rủi ro danh mục tổng hợp (portfolio-level) được tính riêng bằng ma trận hiệp phương sai ở trên."
)

# Nút xoá lịch sử biểu đồ
col_clear, col_info = st.columns([1, 5])
with col_clear:
    if st.button("🗑️ Xoá lịch sử chart", help="Xoá toàn bộ dữ liệu lịch sử của biểu đồ trajectory"):
        st.session_state.history = []
        st.session_state.history_seen_keys = set()
        st.success("Đã xoá lịch sử. Chờ dữ liệu mới...")
with col_info:
    st.caption(f"📊 Đang lưu **{len(st.session_state.history)}** payload | Tối đa {MAX_POINTS_PER_SYMBOL} điểm/mã")

if len(st.session_state.history) > 2:
    _CHART_LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Thời gian",
    )

    # ── Chart 1: VaR 99% ────────────────────────────────────
    df_99 = prepare_var_trajectory_data(st.session_state.history, "var_99")
    if df_99.empty:
        st.warning("Không đủ dữ liệu để vẽ biểu đồ VaR 99% theo thời gian.")
    else:
        fig_99 = px.line(
            df_99,
            x="timestamp",
            y="value_pct",
            color="symbol",
            title="VaR 99% Theo Thời Gian (Asset-level)",
            labels={"timestamp": "Thời gian", "value_pct": "VaR 99% (%)", "symbol": "Mã CK"},
            markers=False,
        )
        fig_99.update_layout(**_CHART_LAYOUT)
        fig_99.update_traces(line=dict(width=2))
        st.plotly_chart(fig_99, use_container_width=True)

    # ── Chart 2: VaR 95% ────────────────────────────────────
    df_95 = prepare_var_trajectory_data(st.session_state.history, "var_95")
    if df_95.empty:
        st.warning("Không đủ dữ liệu để vẽ biểu đồ VaR 95% theo thời gian.")
    else:
        fig_95 = px.line(
            df_95,
            x="timestamp",
            y="value_pct",
            color="symbol",
            title="VaR 95% Theo Thời Gian (Asset-level)",
            labels={"timestamp": "Thời gian", "value_pct": "VaR 95% (%)", "symbol": "Mã CK"},
            markers=False,
        )
        fig_95.update_layout(**_CHART_LAYOUT)
        fig_95.update_traces(line=dict(width=2))
        st.plotly_chart(fig_95, use_container_width=True)

    # ── Chart 3: Volatility Forecast ─────────────────────────
    df_vol = prepare_var_trajectory_data(st.session_state.history, "vol_forecast")
    if df_vol.empty:
        st.warning("Không đủ dữ liệu để vẽ biểu đồ Volatility theo thời gian.")
    else:
        fig_vol = px.line(
            df_vol,
            x="timestamp",
            y="value_pct",
            color="symbol",
            title="Volatility Forecast Theo Thời Gian (Asset-level)",
            labels={"timestamp": "Thời gian", "value_pct": "Volatility (%)", "symbol": "Mã CK"},
            markers=False,
        )
        fig_vol.update_layout(**_CHART_LAYOUT)
        fig_vol.update_traces(line=dict(width=2))
        st.plotly_chart(fig_vol, use_container_width=True)
else:
    st.info("⏳ Đang chờ dữ liệu... (cần ít nhất 3 điểm để vẽ biểu đồ trajectory)")

# ─────────────────────────────────────────────────────────────
# Auto-refresh loop
# ─────────────────────────────────────────────────────────────
time.sleep(1)
st.rerun()
