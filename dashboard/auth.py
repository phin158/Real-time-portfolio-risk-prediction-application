import sqlite3
import os
import hashlib
import datetime
import streamlit as st
from config.settings import get_settings

DB_FILE = "data/users.db"

def init_db():
    """Initialize the SQLite database and create/migrate the users table."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        ''')
        
        # --- Migrations ---
        # Add new columns if they don't exist
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN created_at DATETIME")
        except sqlite3.OperationalError:
            pass
            
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN last_login DATETIME")
        except sqlite3.OperationalError:
            pass
            
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass
            
        conn.commit()
        
    _create_default_admin()

def _create_default_admin():
    """Create default admin from environment variables if it doesn't exist."""
    cfg = get_settings()
    if not cfg.admin_username or not cfg.admin_password:
        return
        
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (cfg.admin_username,))
        if cursor.fetchone() is None:
            salt = os.urandom(16).hex()
            hashed_pw = _get_hash(cfg.admin_password, salt)
            cursor.execute('''
                INSERT INTO users (username, password_hash, salt, role, is_active) 
                VALUES (?, ?, ?, 'admin', 1)
            ''', (cfg.admin_username, hashed_pw, salt))
            conn.commit()

def _get_hash(password: str, salt: str) -> str:
    """Hash password using SHA-256 with a salt."""
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def register_user(username, password) -> bool:
    """Register a new user using SQLite."""
    init_db()
    salt = os.urandom(16).hex()
    hashed_pw = _get_hash(password, salt)
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", 
                           (username, hashed_pw, salt))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # IntegrityError happens if the username already exists (due to UNIQUE constraint)
            return False

def verify_user(username, password) -> dict | None:
    """Verify user credentials against the SQLite database. Returns user info if valid, else None."""
    init_db()
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, password_hash, salt, role, is_active FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        user_id, stored_hash, salt, role, is_active = row
        
        if not is_active:
            return None # User is deactivated
            
        hashed_pw = _get_hash(password, salt)
        
        if hashed_pw == stored_hash:
            # Update last login
            now = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_id))
            conn.commit()
            return {"username": username, "role": role}
        
        return None

def show_auth_page():
    """Render the login/registration UI."""
    st.markdown("""
        <style>
            .auth-title {
                text-align: center;
                font-family: 'Inter', sans-serif;
                margin-bottom: 20px;
                color: #FFFFFF;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="auth-title"><h1>🔐 Hệ thống Quản trị Rủi ro</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])
        
        # --- LOGIN TAB ---
        with tab1:
            st.write("") # Thêm khoảng trắng cho thoáng
            login_username = st.text_input("Tên đăng nhập", key="login_username")
            login_password = st.text_input("Mật khẩu", type="password", key="login_password")
            
            if st.button("Đăng nhập", use_container_width=True, type="primary"):
                if login_username and login_password:
                    user_info = verify_user(login_username, login_password)
                    if user_info:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = user_info["username"]
                        st.session_state["role"] = user_info["role"]
                        st.success("Đăng nhập thành công! Đang chuyển hướng...")
                        st.rerun()
                    else:
                        st.error("Tên đăng nhập không chính xác, sai mật khẩu hoặc tài khoản đã bị khóa.")
                else:
                    st.warning("Vui lòng nhập đầy đủ thông tin.")

        # --- REGISTER TAB ---
        with tab2:
            st.write("") # Thêm khoảng trắng cho thoáng
            reg_username = st.text_input("Tên đăng nhập mới", key="reg_username")
            reg_password = st.text_input("Mật khẩu", type="password", key="reg_password")
            reg_confirm  = st.text_input("Xác nhận mật khẩu", type="password", key="reg_confirm")
            
            if st.button("Đăng ký tài khoản", use_container_width=True):
                if reg_username and reg_password and reg_confirm:
                    if reg_password != reg_confirm:
                        st.error("Mật khẩu xác nhận không khớp.")
                    elif len(reg_password) < 6:
                        st.error("Mật khẩu phải có ít nhất 6 ký tự.")
                    else:
                        if register_user(reg_username, reg_password):
                            st.success("Đăng ký thành công! Vui lòng chuyển sang tab Đăng nhập.")
                        else:
                            st.error("Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.")
                else:
                    st.warning("Vui lòng nhập đầy đủ thông tin.")

def check_authentication():
    """Main entry point to check auth status."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["role"] = "user"
        
    if not st.session_state["logged_in"]:
        show_auth_page()
        st.stop()  # Halt further execution of the dashboard

def is_admin() -> bool:
    """Check if current user is an admin."""
    return st.session_state.get("role") == "admin"

def require_admin():
    """Halt execution if the user is not an admin."""
    if not is_admin():
        st.error("🚫 Bạn không có quyền truy cập chức năng quản trị.")
        st.stop()

def logout():
    """Clear session state and logout."""
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None
    st.rerun()
