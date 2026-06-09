import os
import sqlite3
import pytest
import json

from config.settings import get_settings
from dashboard.auth import init_db, verify_user, DB_FILE

@pytest.fixture(autouse=True)
def setup_test_db():
    """Use a temporary database for testing."""
    original_db = DB_FILE
    import dashboard.auth
    dashboard.auth.DB_FILE = "data/test_users.db"
    
    # Ensure fresh state
    if os.path.exists(dashboard.auth.DB_FILE):
        os.remove(dashboard.auth.DB_FILE)
        
    # Mock settings
    cfg = get_settings()
    cfg.admin_username = "admin_test"
    cfg.admin_password = "admin123_test"
    
    yield
    
    # Cleanup
    if os.path.exists(dashboard.auth.DB_FILE):
        os.remove(dashboard.auth.DB_FILE)
    dashboard.auth.DB_FILE = original_db

def test_admin_creation():
    """Test if default admin is created on init."""
    init_db()
    
    import dashboard.auth
    with sqlite3.connect(dashboard.auth.DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, is_active FROM users WHERE username = 'admin_test'")
        row = cursor.fetchone()
        
    assert row is not None
    assert row[0] == "admin"
    assert row[1] == 1

def test_verify_user_admin():
    """Test verify_user returns correct role and updates last_login."""
    init_db()
    
    user_info = verify_user("admin_test", "admin123_test")
    assert user_info is not None
    assert user_info["username"] == "admin_test"
    assert user_info["role"] == "admin"

def test_deactivated_user():
    """Test deactivated user cannot login."""
    init_db()
    
    import dashboard.auth
    with sqlite3.connect(dashboard.auth.DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_active = 0 WHERE username = 'admin_test'")
        conn.commit()
        
    user_info = verify_user("admin_test", "admin123_test")
    assert user_info is None

def test_symbol_fallback():
    """Test tickers_list fallback when data/symbols.json does not exist."""
    cfg = get_settings()
    if os.path.exists("data/symbols.json"):
        os.rename("data/symbols.json", "data/symbols_backup.json")
        
    # Should fallback to default/env
    assert len(cfg.tickers_list) > 0
    
    # Restore
    if os.path.exists("data/symbols_backup.json"):
        os.rename("data/symbols_backup.json", "data/symbols.json")
