# Real-Time Portfolio Risk Prediction Application

> **Stack:** Python · FastAPI · Apache Kafka · Redis Pub/Sub · WebSocket · PyTorch TFT · Streamlit · MLflow · SQLite

---

## Tổng quan hệ thống

Ứng dụng dự đoán rủi ro danh mục đầu tư theo thời gian thực, sử dụng **Temporal Fusion Transformer (TFT)** để dự đoán VaR (Value at Risk), CVaR (Expected Shortfall) và biến động theo phút. Giao diện được tích hợp hệ thống phân quyền (Authentication/RBAC) và quản lý cấu hình động.

```
yfinance replay ──► Kafka Producer ──► Kafka Consumer
                                              │
                                    FeatureEngineer (12 features)
                                              │
                          ┌───────────────────┴──────────────────────┐
                          │ TFT Model (if checkpoint exists)         │
                          │ OR Baseline (EWMA + Historical VaR)      │
                          └───────────────────┬──────────────────────┘
                                              │
                                    PortfolioRiskAggregator
                                    (w.T @ Σ @ w — covariance VaR)
                                              │
                               Redis Pub/Sub ─┤─ WebSocket ─► Dashboard ◄── SQLite (Users & Roles)
                                              │
                                     REST API /api/portfolio
```

---

## Feature Set (12 features)

| # | Tên feature | Công thức | Ý nghĩa |
|---|---|---|---|
| 0 | `log_return` | ln(P_t / P_{t-1}) | Log return 1-phút |
| 1 | `vol_30` | σ(r, 30) × √(252×390) | Annualised volatility 30 bars |
| 2 | `vol_60` | σ(r, 60) × √(252×390) | Annualised volatility 60 bars |
| 3 | `vol_390` | σ(r, 390) × √(252×390) | Annualised volatility 1 ngày |
| 4 | `rsi_14` | RSI(14) Wilder | RSI 14 bars |
| 5 | `macd_line` | EMA(12) − EMA(26) | MACD line |
| 6 | `macd_signal` | EMA(9, MACD) | MACD signal |
| 7 | `macd_hist` | MACD − Signal | MACD histogram |
| 8 | `zscore_30` | (P_t − μ) / σ | Z-score giá 30 bars |
| 9 | `volume_change` | (V_t − V_{t-1}) / V_{t-1} | Tốc độ thay đổi khối lượng |
| 10 | `volume_zscore_30` | (V_t − μ_V) / σ_V | Z-score khối lượng 30 bars |
| 11 | `dollar_volume` | Close × Volume | Dollar volume (proxy thanh khoản) |

---

## Portfolio Risk Formula

```
sigma²_p = w.T @ Sigma @ w           # Covariance-based portfolio variance
sigma_p  = sqrt(sigma²_p)             # Portfolio volatility

VaR_95 (Parametric) = max(0, -(mu_p + z_05 * sigma_p))   z_05 = -1.6449
VaR_99 (Parametric) = max(0, -(mu_p + z_01 * sigma_p))   z_01 = -2.3263

VaR_95 (Historical) = -quantile(portfolio_returns, 0.05)
VaR_99 (Historical) = -quantile(portfolio_returns, 0.01)

CVaR_95 = -mean(returns[returns <= -VaR_95])
CVaR_99 = -mean(returns[returns <= -VaR_99])
```

---

## Cài đặt và chạy

### Yêu cầu

```bash
# Python 3.12+, Docker
pip install -r requirements.txt
```

### Khởi động infrastructure

```bash
docker-compose up -d   # Khởi chạy Kafka + Zookeeper + Redis
```

### Chạy hệ thống toàn diện

Sử dụng các shell script để chạy từng thành phần:

```bash
# Terminal 1: MLflow Tracking Server (Cổng 5001)
bash run_mlflow.sh

# Terminal 2: API Backend & WebSocket
bash run_backend.sh

# Terminal 3: Data Producer (Kafka streaming)
bash run_producer.sh

# Terminal 4: Dashboard UI (Streamlit)
bash run_dashboard.sh
```

---

## Giao diện Dashboard & Phân quyền

Hệ thống có 2 chế độ hiển thị dựa trên quyền người dùng, lưu trữ trong `data/users.db`:

1. **User Dashboard**: Bảng điều khiển chính xem VaR, CVaR, Correlation Heatmap và các chỉ số tài sản theo thời gian thực.
2. **Admin Dashboard**: Cung cấp các công cụ quản trị (Quản lý User, Trạng thái Hệ thống, Theo dõi Checkpoint, Đọc báo cáo Backtesting, Cấu hình danh sách mã cổ phiếu động `data/symbols.json`).

> Lưu ý: Mã cổ phiếu hiện được cấu hình động qua Admin UI, ghi đè lên file `.env` mặc định.

---

## MLflow & Training

Hệ thống tích hợp MLflow để theo dõi metrics trong quá trình training và backtesting.

### Train model TFT

```bash
# Tải dữ liệu và train (time-based split tự động)
python scripts/train_tft.py --period 30d --epochs 20

# Kết quả lưu tại:
#   model/checkpoints/tft_best.pt          (checkpoint)
#   model/checkpoints/tft_best_metadata.json (training info)
```

Kết quả train cũng được tự động log lên MLflow Tracking Server tại `http://localhost:5001`.

---

## Fallback tự động

Nếu `model/checkpoints/tft_best.pt` **chưa tồn tại**, hệ thống tự động dùng:
- **EWMA Volatility** (RiskMetrics, λ=0.94) cho per-symbol dự đoán
- **Historical VaR** (empirical quantile) cho fallback VaR
- Portfolio-level VaR dùng covariance matrix (không đổi)

---

## API Endpoints

### `POST /api/portfolio`
Tính toán VaR, CVaR và biến động cho danh mục.
```json
{
  "weights": {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.1, "NVDA": 0.1},
  "covariance_window": 390
}
```

### `GET /api/metrics`
Per-symbol predictions (TFT hoặc baseline).

### `WebSocket ws://localhost:8000/ws/risk-stream`
Stream real-time predictions từ Redis Pub/Sub.

---

## Backtesting (Basel III)

```bash
python scripts/backtest_var.py
```
Output bao gồm kết quả Kupiec POF Test và phân loại vùng Basel III Traffic Light. Báo cáo được hiển thị trực tiếp trên giao diện Admin Dashboard và log lên MLflow.

---

## Test Suite

```bash
pytest tests/ -v
# 121 tests — tất cả PASS ✅
```

| Module | Tests | Nội dung |
|---|---|---|
| `test_portfolio_risk.py` | 26 | Covariance VaR, Historical VaR, CVaR, reliability flags |
| `test_predictor.py` | 14 | Checkpoint guard, NaN guard, baseline fallback |
| `test_feature_engineer.py` | 48 | Feature computation, warm-up, tensor shape |
| `test_producer.py` | 9 | Kafka serialization |
| `test_validator.py` | 16 | Tick validation, outlier detection |
| `test_api.py` | 3 | API endpoints |
| `test_admin.py` | 5 | Authentication, RBAC, Admin logic |

---

## Cấu trúc thư mục hiện tại

```
portfolio_risk_app/
├── api/
│   ├── main.py
│   ├── routers/
│   │   ├── portfolio.py      # POST /api/portfolio (covariance VaR)
│   │   └── websocket.py      # WS /ws/risk-stream
│   └── services/
│       └── risk_service.py   # Orchestrator + baseline fallback
├── data_pipeline/
│   ├── producer.py           # Kafka Producer
│   ├── consumer.py           # Kafka Consumer
│   ├── schemas.py
│   └── validator.py
├── feature_engineering/
│   ├── engineer.py           # FeatureEngineer (12 features)
│   └── indicators.py         # Pure-function indicators
├── model/
│   ├── tft.py                # TFT architecture
│   ├── predictor.py          # RiskPredictor với checkpoint guard
│   └── baseline.py           # EWMA + Historical VaR fallback
├── portfolio/
│   └── risk_aggregator.py    # Covariance VaR, CVaR, Historical VaR
├── scripts/
│   ├── train_tft.py          # Training pipeline (MLflow integrated)
│   └── backtest_var.py       # Kupiec test + Basel III (MLflow integrated)
├── dashboard/
│   ├── app.py                # Streamlit user dashboard
│   ├── auth.py               # Xác thực người dùng (SQLite)
│   └── admin.py              # Trang quản trị Admin
├── data/
│   ├── users.db              # SQLite Database (User accounts)
│   └── symbols.json          # Cấu hình mã cổ phiếu động
├── tests/                    # 121 unit tests
├── config/
│   └── settings.py
├── run_*.sh                  # Shell scripts khởi động
└── .env
```
