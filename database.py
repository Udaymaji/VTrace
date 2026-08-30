import sqlite3
import hashlib
import os
from datetime import datetime, timedelta

DB_PATH = "vitaltrace.db"

def get_db_connection():
    """Returns a row-factory SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    """SHA-256 password hashing."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_db():
    """Initializes SQLite database schema and seeds initial data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'doctor',
        full_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Patients Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        phone TEXT,
        email TEXT,
        medical_history TEXT,
        emergency_contact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        monitoring_status TEXT DEFAULT 'INACTIVE'
    );
    """)

    # 3. Vitals Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vitals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        heart_rate REAL NOT NULL,
        spo2 REAL NOT NULL,
        temperature REAL NOT NULL,
        respiratory_rate REAL NOT NULL,
        systolic_bp REAL NOT NULL,
        diastolic_bp REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
    );
    """)

    # 4. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        prediction TEXT NOT NULL,
        risk_score REAL NOT NULL,
        confidence REAL NOT NULL,
        prob_normal REAL DEFAULT 0,
        prob_warning REAL DEFAULT 0,
        prob_critical REAL DEFAULT 0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
    );
    """)

    # 5. Alerts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        message TEXT NOT NULL,
        risk_score REAL NOT NULL,
        confidence REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'ACTIVE',
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
    );
    """)

    # Seed Default Accounts if missing
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO users (username, password_hash, role, full_name) VALUES
        ('admin', ?, 'admin', 'Dr. Meera Desai (Chief Medical Officer)'),
        ('doctor', ?, 'doctor', 'Dr. Rajesh Sharma (Lead Intensivist)')
        """, (hash_password('admin123'), hash_password('doctor123')))
        print("[Database] Seeded default users: admin / doctor")

    # Seed Demo Patients if none exist
    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        demo_patients = [
            ("VT-1001", "Aarav Patel", 62, "Male", "9876543210", "aarav.p@med.org", "Post-op CABG Day 2, Mild Hypertension", "Sneha Patel (Spouse) - 9876543211"),
            ("VT-1002", "Priya Singh", 54, "Female", "9876543212", "priya.s@univ.edu", "Type 2 Diabetes, Chronic Kidney Disease Stage 2", "Rahul Singh (Son) - 9876543213"),
            ("VT-1003", "Kavya Menon", 29, "Female", "9876543214", "kavya.menon@tech.io", "Asthma, Recovering Viral Pneumonia", "Arjun Menon (Brother) - 9876543215"),
            ("VT-1004", "Vikram Reddy", 71, "Male", "9876543216", "v.reddy@home.net", "Atrial Fibrillation, Anticoagulant Therapy", "Anjali Reddy (Wife) - 9876543217")
        ]
        cursor.executemany("""
        INSERT INTO patients (patient_id, full_name, age, gender, phone, email, medical_history, emergency_contact, monitoring_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'INACTIVE')
        """, demo_patients)
        print("[Database] Seeded 4 initial ICU demo patients")

    conn.commit()
    conn.close()

# ----------------- User Authentication Queries ----------------- #

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, full_name FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_username(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, full_name FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

# ----------------- Patient Queries ----------------- #

def get_all_patients():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT p.*,
           (SELECT prediction FROM predictions WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_prediction,
           (SELECT risk_score FROM predictions WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_risk_score,
           (SELECT confidence FROM predictions WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_confidence,
           (SELECT heart_rate FROM vitals WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_hr,
           (SELECT spo2 FROM vitals WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_spo2,
           (SELECT temperature FROM vitals WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_temp,
           (SELECT respiratory_rate FROM vitals WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_rr,
           (SELECT systolic_bp FROM vitals WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_sys,
           (SELECT diastolic_bp FROM vitals WHERE patient_id = p.patient_id ORDER BY id DESC LIMIT 1) as latest_dia
    FROM patients p
    ORDER BY p.created_at DESC
    """)
    rows = cursor.fetchall()
    patients = [dict(r) for r in rows]
    conn.close()
    return patients

def get_patient_by_id(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ? OR id = ?", (patient_id, patient_id))
    patient = cursor.fetchone()
    conn.close()
    return dict(patient) if patient else None

def create_patient(full_name, age, gender, phone="", email="", medical_history="", emergency_contact=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Auto generate distinct Patient ID: VT-XXXX
    cursor.execute("SELECT MAX(id) FROM patients")
    max_id = cursor.fetchone()[0] or 1000
    new_pid = f"VT-{max_id + 1001}"
    
    cursor.execute("""
    INSERT INTO patients (patient_id, full_name, age, gender, phone, email, medical_history, emergency_contact, monitoring_status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'INACTIVE')
    """, (new_pid, full_name, int(age), gender, phone, email, medical_history, emergency_contact))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return get_patient_by_id(new_pid)

def update_patient(patient_id, full_name, age, gender, phone="", email="", medical_history="", emergency_contact=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE patients
    SET full_name = ?, age = ?, gender = ?, phone = ?, email = ?, medical_history = ?, emergency_contact = ?
    WHERE patient_id = ?
    """, (full_name, int(age), gender, phone, email, medical_history, emergency_contact, patient_id))
    conn.commit()
    conn.close()
    return get_patient_by_id(patient_id)

def delete_patient(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM vitals WHERE patient_id = ?", (patient_id,))
    cursor.execute("DELETE FROM predictions WHERE patient_id = ?", (patient_id,))
    cursor.execute("DELETE FROM alerts WHERE patient_id = ?", (patient_id,))
    cursor.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return True

def set_patient_monitoring_status(patient_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE patients SET monitoring_status = ? WHERE patient_id = ?", (status, patient_id))
    conn.commit()
    conn.close()

# ----------------- Vitals & Predictions Storage ----------------- #

def record_vital_reading(patient_id, hr, spo2, temp, rr, sys_bp, dia_bp):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO vitals (patient_id, heart_rate, spo2, temperature, respiratory_rate, systolic_bp, diastolic_bp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, float(hr), float(spo2), float(temp), float(rr), float(sys_bp), float(dia_bp)))
    conn.commit()
    conn.close()

def record_prediction(patient_id, prediction, risk_score, confidence, prob_normal=0.0, prob_warning=0.0, prob_critical=0.0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (patient_id, prediction, risk_score, confidence, prob_normal, prob_warning, prob_critical)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, prediction, float(risk_score), float(confidence), float(prob_normal), float(prob_warning), float(prob_critical)))
    conn.commit()
    conn.close()

def get_recent_vitals(patient_id, limit=30):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM vitals WHERE patient_id = ? ORDER BY id DESC LIMIT ?
    """, (patient_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

def get_patient_history(patient_id, time_filter="all"):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    where_clause = "WHERE v.patient_id = ?"
    params = [patient_id]
    
    if time_filter == "today":
        where_clause += " AND v.timestamp >= date('now', 'start of day')"
    elif time_filter == "24h":
        where_clause += " AND v.timestamp >= datetime('now', '-24 hours')"
    elif time_filter == "7d":
        where_clause += " AND v.timestamp >= datetime('now', '-7 days')"

    query = f"""
    SELECT v.id, v.patient_id, v.heart_rate, v.spo2, v.temperature, v.respiratory_rate,
           v.systolic_bp, v.diastolic_bp, v.timestamp,
           p.prediction, p.risk_score, p.confidence, p.prob_normal, p.prob_warning, p.prob_critical
    FROM vitals v
    LEFT JOIN predictions p ON v.patient_id = p.patient_id AND ABS(strftime('%s', v.timestamp) - strftime('%s', p.timestamp)) < 3
    {where_clause}
    ORDER BY v.timestamp DESC
    LIMIT 500
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ----------------- Alerts Queries ----------------- #

def create_alert(patient_id, alert_type, message, risk_score, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO alerts (patient_id, alert_type, message, risk_score, confidence, status)
    VALUES (?, ?, ?, ?, ?, 'ACTIVE')
    """, (patient_id, alert_type, message, float(risk_score), float(confidence)))
    conn.commit()
    conn.close()

def get_alerts(status_filter=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT a.*, p.full_name as patient_name
    FROM alerts a
    JOIN patients p ON a.patient_id = p.patient_id
    """
    params = []
    if status_filter and status_filter.upper() != 'ALL':
        if status_filter.upper() in ['WARNING', 'CRITICAL']:
            query += " WHERE a.alert_type = ?"
            params.append(status_filter.upper())
        elif status_filter.upper() == 'RESOLVED':
            query += " WHERE a.status = 'RESOLVED'"
        elif status_filter.upper() == 'ACTIVE':
            query += " WHERE a.status = 'ACTIVE'"
            
    query += " ORDER BY a.timestamp DESC LIMIT 200"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def resolve_alert(alert_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET status = 'RESOLVED' WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()
    return True

# ----------------- System Stats ----------------- #

def get_dashboard_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patients WHERE monitoring_status = 'ACTIVE'")
    active_monitoring = cursor.fetchone()[0]
    
    # Active patient counts by latest prediction
    cursor.execute("""
    SELECT 
        SUM(CASE WHEN p.prediction = 'NORMAL' THEN 1 ELSE 0 END) as normal_count,
        SUM(CASE WHEN p.prediction = 'WARNING' THEN 1 ELSE 0 END) as warning_count,
        SUM(CASE WHEN p.prediction = 'CRITICAL' THEN 1 ELSE 0 END) as critical_count
    FROM (
        SELECT patient_id, prediction FROM predictions GROUP BY patient_id HAVING MAX(id)
    ) p
    """)
    row = cursor.fetchone()
    normal_count = (row[0] or 0) if row else 0
    warning_count = (row[1] or 0) if row else 0
    critical_count = (row[2] or 0) if row else 0
    
    # If some patients have no predictions yet, treat as Normal
    if (normal_count + warning_count + critical_count) < total_patients:
        normal_count = total_patients - (warning_count + critical_count)

    cursor.execute("SELECT COUNT(*) FROM alerts WHERE status = 'ACTIVE'")
    active_alerts = cursor.fetchone()[0]

    conn.close()
    return {
        'total_patients': total_patients,
        'active_monitoring': active_monitoring,
        'normal_count': normal_count,
        'warning_count': warning_count,
        'critical_count': critical_count,
        'active_alerts': active_alerts
    }
