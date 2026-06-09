#!/bin/bash
# run_mlflow.sh — Start MLflow Tracking Server (Phase 7)
#
# Chạy MLflow Tracking Server trực tiếp bằng virtualenv của project,
# nhất quán với run_backend.sh, run_producer.sh, run_dashboard.sh.
#
# Usage:
#   bash run_mlflow.sh
#
# Sau khi chạy, truy cập UI tại: http://localhost:5001
# (Port 5001 thay vì 5000 vì macOS AirPlay Receiver chiếm port 5000)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Kiểm tra virtualenv ───────────────────────────────────────────────────────
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Không tìm thấy .venv — chạy: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# ── Kiểm tra mlflow đã cài chưa ──────────────────────────────────────────────
if ! .venv/bin/python -c "import mlflow" 2>/dev/null; then
    echo "📦 MLflow chưa được cài, đang cài..."
    .venv/bin/pip install "mlflow>=2.13.0" --quiet
fi

# ── Tạo thư mục lưu trữ ──────────────────────────────────────────────────────
mkdir -p mlflow_store/artifacts

echo "========================================"
echo "  MLflow Tracking Server"
echo "  UI:       http://localhost:5001"
echo "  Backend:  sqlite:///mlflow_store/mlflow.db"
echo "  Artifacts: ./mlflow_store/artifacts"
echo "========================================"

# ── Khởi động server ─────────────────────────────────────────────────────────
.venv/bin/mlflow server \
    --host 127.0.0.1 \
    --port 5001 \
    --backend-store-uri "sqlite:///$(pwd)/mlflow_store/mlflow.db" \
    --default-artifact-root "$(pwd)/mlflow_store/artifacts" \
    --serve-artifacts
