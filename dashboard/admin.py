import json
import os
import sqlite3
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

from dashboard.auth import require_admin, DB_FILE
from config.settings import get_settings

def get_all_users():
    """Fetch all users from SQLite."""
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("SELECT id, username, role, created_at, last_login, is_active FROM users", conn)
    return df

def update_user_status(user_id, is_active):
    """Update user is_active status."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (is_active, user_id))
        conn.commit()

def update_user_role(user_id, role):
    """Update user role."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()

def delete_user(user_id):
    """Delete a user."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

def check_fastapi_health(api_url):
    """Check FastAPI health endpoint."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=2)
        return "Running" if resp.status_code == 200 else "Error"
    except Exception:
        return "Không xác định (Not Reachable)"

def check_redis_status(redis_host, redis_port):
    """Check Redis using ping."""
    try:
        import redis
        client = redis.Redis(host=redis_host, port=redis_port, socket_timeout=2)
        return "Running" if client.ping() else "Error"
    except Exception:
        return "Không xác định (Not Reachable)"

def show_user_management():
    st.subheader("👥 Quản lý Người dùng")
    
    users_df = get_all_users()
    
    st.dataframe(users_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### Thao tác với Người dùng")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Đổi quyền (Role)**")
        user_to_role = st.selectbox("Chọn User ID", users_df['id'].tolist(), key="role_user_id")
        new_role = st.selectbox("Quyền mới", ["user", "admin"])
        if st.button("Cập nhật quyền", type="primary"):
            # Validation: ensure at least one active admin remains if changing from admin to user
            target_user = users_df[users_df['id'] == user_to_role].iloc[0]
            if target_user['role'] == 'admin' and new_role == 'user':
                active_admins = users_df[(users_df['role'] == 'admin') & (users_df['is_active'] == 1)]
                if len(active_admins) <= 1:
                    st.error("Không thể hủy quyền Admin cuối cùng!")
                    st.stop()
            update_user_role(user_to_role, new_role)
            st.success("Cập nhật quyền thành công!")
            st.rerun()

    with col2:
        st.markdown("**Trạng thái (Active/Deactive)**")
        user_to_status = st.selectbox("Chọn User ID", users_df['id'].tolist(), key="status_user_id")
        new_status = st.radio("Trạng thái", [1, 0], format_func=lambda x: "Hoạt động (1)" if x == 1 else "Vô hiệu hóa (0)")
        if st.button("Cập nhật trạng thái"):
            target_user = users_df[users_df['id'] == user_to_status].iloc[0]
            if target_user['username'] == st.session_state.get('username') and new_status == 0:
                st.error("Bạn không thể tự vô hiệu hóa chính mình!")
            else:
                update_user_status(user_to_status, new_status)
                st.success("Cập nhật trạng thái thành công!")
                st.rerun()

    with col3:
        st.markdown("**Xóa Người dùng**")
        user_to_delete = st.selectbox("Chọn User ID", users_df['id'].tolist(), key="delete_user_id")
        if st.button("Xóa tài khoản", type="secondary"):
            target_user = users_df[users_df['id'] == user_to_delete].iloc[0]
            if target_user['username'] == st.session_state.get('username'):
                st.error("Bạn không thể tự xóa chính mình!")
            elif target_user['role'] == 'admin':
                active_admins = users_df[(users_df['role'] == 'admin') & (users_df['is_active'] == 1)]
                if len(active_admins) <= 1:
                    st.error("Không thể xóa Admin cuối cùng!")
                    st.stop()
                else:
                    delete_user(user_to_delete)
                    st.success("Đã xóa người dùng.")
                    st.rerun()
            else:
                delete_user(user_to_delete)
                st.success("Đã xóa người dùng.")
                st.rerun()

def show_system_status():
    st.subheader("🖥️ Trạng thái Hệ thống")
    cfg = get_settings()
    
    fastapi_status = check_fastapi_health(f"http://localhost:{cfg.api_port}")
    redis_status = check_redis_status(cfg.redis_host, cfg.redis_port)
    kafka_status = "Không xác định (Not directly checkable via Python)"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("FastAPI Backend", fastapi_status)
    col2.metric("Redis Pub/Sub", redis_status)
    col3.metric("Kafka Broker", kafka_status)
    
    st.markdown("---")
    st.markdown("#### Cấu hình Hệ thống hiện tại")
    
    st.write(f"- **Kafka Brokers**: `{cfg.kafka_bootstrap_servers}`")
    st.write(f"- **Kafka Topic**: `{cfg.kafka_topic_market_data}`")
    st.write(f"- **Redis Channel**: `{cfg.redis_channel}`")
    st.write(f"- **Số lượng mã cổ phiếu theo dõi**: {len(cfg.tickers_list)}")
    st.write(f"- **Danh sách mã**: {', '.join(cfg.tickers_list)}")
    
def show_model_checkpoint():
    st.subheader("🧠 Trạng thái Model & Checkpoint")
    cfg = get_settings()
    
    ckpt_path = cfg.model_checkpoint_path
    ckpt_exists = os.path.exists(ckpt_path)
    
    metadata_path = "model/checkpoints/tft_best_metadata.json"
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass
            
    if ckpt_exists:
        st.success(f"✅ Checkpoint tồn tại tại: `{ckpt_path}`")
    else:
        st.error(f"❌ Checkpoint chưa tồn tại tại: `{ckpt_path}`. Hệ thống sẽ dùng baseline EWMA + Historical VaR.")
        
    st.markdown("#### Chi tiết cấu hình TFT")
    st.write(f"- **Hidden Size (cấu hình hiện tại)**: {cfg.hidden_size}")
    if metadata:
        st.json(metadata)
    else:
        st.info("Không tìm thấy file metadata của model.")

def show_backtesting_summary():
    st.subheader("📉 Kết quả VaR Backtesting")
    
    # Try different possible paths
    paths = [
        "model/checkpoints/backtest_summary.json",
        "model/checkpoints/backtest_results.json",
        "outputs/backtest_summary.json",
        "reports/backtest_summary.json",
        "backtest_results.json"
    ]
    
    summary_path = None
    for p in paths:
        if os.path.exists(p):
            summary_path = p
            break
            
    if not summary_path:
        st.warning("⚠️ Chưa tìm thấy file kết quả Backtesting.")
        st.info("Hướng dẫn: Để chạy backtesting, hãy chạy lệnh sau trong terminal:")
        st.code("python scripts/backtest_var.py")
        return
        
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
            
        st.success(f"Đã tải báo cáo từ: `{summary_path}`")
        
        st.markdown("""
        **Giải thích:**
        - **Violation rate** là tỷ lệ số lần lỗ thực tế vượt quá ngưỡng VaR dự báo.
        - **VaR 95%** kỳ vọng vi phạm khoảng 5%.
        - **VaR 99%** kỳ vọng vi phạm khoảng 1%.
        """)
        
        n_obs = data.get("number_of_observations", data.get("total_observations", "N/A"))
        st.write(f"**Số lượng quan sát:** {n_obs}")
        
        # We need to flexibly parse the backtest json as structure might vary
        # Typically it's structured by symbol -> var_95_violation_rate etc.
        st.json(data)
            
    except Exception as e:
        st.error(f"Lỗi khi đọc file Backtesting: {e}")

def show_symbol_config():
    st.subheader("⚙️ Cấu hình Mã Cổ phiếu (Symbols)")
    cfg = get_settings()
    current_symbols = cfg.tickers_list
    
    st.info("⚠️ Thay đổi mã cổ phiếu sẽ có hiệu lực sau khi khởi động lại producer/backend.")
    
    st.write("**Danh sách mã cổ phiếu hiện tại:**")
    st.write(", ".join(current_symbols))
    
    st.markdown("---")
    new_symbol = st.text_input("Thêm mã mới (VD: TSLA)").strip().upper()
    if st.button("Thêm Mã"):
        if not new_symbol:
            st.warning("Mã cổ phiếu không được để trống!")
        elif new_symbol in current_symbols:
            st.warning("Mã cổ phiếu đã tồn tại!")
        else:
            current_symbols.append(new_symbol)
            _save_symbols(current_symbols)
            st.success(f"Đã thêm {new_symbol}. Vui lòng restart system!")
            st.rerun()
            
    st.markdown("---")
    remove_symbol = st.selectbox("Chọn mã để xóa", current_symbols)
    if st.button("Xóa Mã", type="secondary"):
        if len(current_symbols) <= 1:
            st.error("Phải giữ lại ít nhất 1 mã cổ phiếu!")
        else:
            current_symbols.remove(remove_symbol)
            _save_symbols(current_symbols)
            st.success(f"Đã xóa {remove_symbol}. Vui lòng restart system!")
            st.rerun()
            
    st.markdown("---")
    if st.button("Reset về Mặc định"):
        default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        _save_symbols(default_symbols)
        st.success("Đã reset danh sách mã cổ phiếu. Vui lòng restart system!")
        st.rerun()

def _save_symbols(symbols):
    """Save symbols to data/symbols.json"""
    os.makedirs("data", exist_ok=True)
    with open("data/symbols.json", "w") as f:
        json.dump(symbols, f)

def show_admin_dashboard():
    require_admin()
    
    st.title("🛡️ Bảng Điều khiển Quản trị (Admin Dashboard)")
    
    tabs = st.tabs([
        "Quản lý User", 
        "Trạng thái Hệ thống", 
        "Model & Checkpoint", 
        "Backtesting Summary", 
        "Cấu hình Symbols"
    ])
    
    with tabs[0]:
        show_user_management()
    with tabs[1]:
        show_system_status()
    with tabs[2]:
        show_model_checkpoint()
    with tabs[3]:
        show_backtesting_summary()
    with tabs[4]:
        show_symbol_config()
