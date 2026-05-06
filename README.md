# Real-Time Portfolio Risk Prediction

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg)
![Kafka](https://img.shields.io/badge/Kafka-3.7.0-black.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)

Hệ thống **Real-Time Portfolio Risk Prediction** là một ứng dụng tài chính định lượng cung cấp khả năng phân tích và dự báo rủi ro danh mục đầu tư theo thời gian thực. Hệ thống sử dụng kiến trúc Event-Driven kết hợp với mô hình Học sâu (Deep Learning) **Temporal Fusion Transformer (TFT)** để dự báo Value at Risk (VaR) và độ biến động (Volatility) với độ trễ siêu thấp.

---

## 🏗 Kiến trúc Hệ thống (System Architecture)

Dữ liệu di chuyển xuyên suốt hệ thống theo một Data Pipeline theo thời gian thực khép kín:

1. **Data Ingestion (`yfinance` ➔ Kafka Producer)**: Dữ liệu giá cổ phiếu (OHLCV) được stream liên tục theo từng phút (1-min bar) giả lập từ yfinance và đẩy vào Apache Kafka.
2. **Streaming Engine (Kafka KRaft)**: Quản lý message queue đảm bảo throughput cao và khả năng mở rộng (scalability).
3. **Data Validation & Feature Engineering**: Kafka Consumer kéo dữ liệu, làm sạch (loại bỏ outliers, NaN) và tính toán 9 đặc trưng tài chính (Log Return, Rolling Volatility, RSI, MACD...) qua cửa sổ trượt (sliding window).
4. **Deep Learning Inference (PyTorch TFT Model)**: Tensor dữ liệu được đưa vào mô hình Temporal Fusion Transformer (đã huấn luyện) để nội suy (infer) dự báo biến động và lượng tử hoá rủi ro (VaR 95%, 99%).
5. **Real-Time Broadcasting (FastAPI ➔ Redis Pub/Sub)**: Kết quả dự báo được publish lên Redis và đẩy thẳng qua WebSocket.
6. **Frontend Dashboard (Streamlit)**: Giao diện nhận dữ liệu từ WebSocket qua Redis, vẽ biểu đồ động và cho phép người dùng thay đổi tỷ trọng (weights) danh mục để tính rủi ro gộp tức thời.

---

## 🛠 Tech Stack

- **Data Pipeline & Streaming**: Apache Kafka (KRaft mode - Zookeeperless)
- **Feature Engineering**: Pandas, Numpy (vectorized stateless computations)
- **Machine Learning**: PyTorch (Temporal Fusion Transformer, Quantile Loss)
- **Backend API & Websocket**: FastAPI, Uvicorn, Redis Pub/Sub, websockets
- **Frontend / UI**: Streamlit, Plotly (Dynamic Charts)
- **Infrastructure**: Docker Desktop, Docker Compose

---

## 📂 Cấu trúc Thư mục (Project Structure)

```text
portfolio_risk_app/
├── api/                   # FastAPI Backend (REST & WebSocket)
│   ├── main.py            # Entry point & Lifespan context
│   ├── routers/           # REST endpoints & WS handler
│   └── services/          # RiskService orchestrating DL model & Redis
├── config/                # Quản lý thiết lập bằng Pydantic BaseSettings
├── dashboard/             # Streamlit Frontend
│   └── app.py             # Giao diện UI/UX thời gian thực
├── data_pipeline/         # Data Ingestion & Streaming
│   ├── consumer.py        # Kafka Consumer
│   ├── producer.py        # Mock Real-time Data Producer
│   ├── schemas.py         # Pydantic schemas cho dữ liệu Tick
│   └── validator.py       # Thuật toán bắt Outlier và làm sạch data
├── feature_engineering/   # Xử lý đặc trưng tài chính (Indicators)
├── model/                 # Kiến trúc Temporal Fusion Transformer
├── scripts/               # Scripts chạy Offline (Data Generator)
├── tests/                 # Unit tests & Integration tests (Pytest)
├── docker-compose.yml     # Khởi tạo Kafka và Redis
└── requirements.txt       # Dependencies
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy dự án (How to run)

### 1. Yêu cầu hệ thống
- macOS / Linux / Windows (WSL2)
- Python 3.12+
- Docker & Docker Compose

### 2. Thiết lập Môi trường (Setup Environment)
Tạo môi trường ảo và cài đặt thư viện:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cấu hình biến môi trường:
```bash
cp .env.example .env
# Chỉnh sửa file .env nếu cần (mặc định đã sẵn sàng cho localhost)
```

### 3. Khởi động Infrastructure (Kafka & Redis)
Chạy nền dịch vụ qua Docker Compose:
```bash
docker compose up -d
```

### 4. Khởi chạy Hệ thống (Mở 3 Terminal riêng biệt)

Chúng tôi đã viết sẵn 3 scripts tự động kích hoạt môi trường ảo. 

**Terminal 1: Chạy Backend (FastAPI)**
```bash
./run_backend.sh
# Hoặc thủ công: uvicorn api.main:app --reload --port 8000
```

**Terminal 2: Chạy Data Producer (Phát sóng dữ liệu)**
```bash
./run_producer.sh
# Hoặc thủ công: PYTHONPATH=. python data_pipeline/producer.py
```

**Terminal 3: Chạy Giao diện Streamlit**
```bash
./run_dashboard.sh
# Hoặc thủ công: PYTHONPATH=. streamlit run dashboard/app.py
```

Trải nghiệm bảng điều khiển trực tiếp tại: **http://localhost:8501**
