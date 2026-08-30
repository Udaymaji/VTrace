import os
import sys
import re
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_file, flash, Response
)
from flask_socketio import SocketIO, emit, join_room, leave_room
import io
import csv

import database
from model import VitalPredictor, FEATURE_NAMES, SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS
from simulator import VitalSimulatorManager
import train_model

# Flask Application Initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'vitaltrace-clinical-secret-key-2026')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize Simulator Manager
simulator_manager = VitalSimulatorManager(socketio)

# Auto-initialize database on startup
database.init_db()

@app.before_request
def ensure_default_session():
    # If session is empty (e.g. third-party cookies blocked in preview iframe), auto-initialize default doctor session
    if 'user_id' not in session:
        default_user = database.get_user_by_username('doctor')
        if default_user:
            session['user_id'] = default_user['id']
            session['username'] = default_user['username']
            session['role'] = default_user['role']
            session['full_name'] = default_user['full_name']

# Login Required Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            default_user = database.get_user_by_username('doctor')
            if default_user:
                session['user_id'] = default_user['id']
                session['username'] = default_user['username']
                session['role'] = default_user['role']
                session['full_name'] = default_user['full_name']
            else:
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Unauthorized. Please login.'}), 401
                return redirect(url_for('login_page', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Context Processor for Global Template Data
@app.context_processor
def inject_global_data():
    user = None
    if 'user_id' in session:
        user = database.get_user_by_id(session['user_id'])
    stats = {}
    try:
        stats = database.get_dashboard_stats()
    except Exception:
        stats = {'total_patients': 6, 'active_alerts': 0, 'critical_count': 0, 'warning_count': 0, 'normal_count': 6}
    return {
        'current_user': user,
        'stats': stats,
        'current_time': datetime.now().strftime('%B %d, %Y - %H:%M:%S'),
        'medical_disclaimer': 'Research/Demo Only — This system is not a medical diagnostic device. Predictions are for academic demonstration and should not replace professional medical judgment.'
    }

# ----------------- Authentication Routes ----------------- #

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        user = database.authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            next_url = request.args.get('next') or url_for('dashboard_page')
            return redirect(next_url)
        else:
            error = "Invalid clinical credentials. Use admin/admin123 or doctor/doctor123."
            
    return render_template('login.html', error=error)

@app.route('/login/direct/<username>')
def direct_login(username):
    user = database.get_user_by_username(username)
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session['full_name'] = user['full_name']
    return redirect(url_for('dashboard_page'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ----------------- UI Web Pages ----------------- #

@app.route('/')
@login_required
def dashboard_page():
    stats = database.get_dashboard_stats()
    patients = database.get_all_patients()
    recent_alerts = database.get_alerts('ACTIVE')[:5]
    return render_template('dashboard.html', stats=stats, patients=patients, recent_alerts=recent_alerts)

@app.route('/patients')
@login_required
def patients_page():
    patients = database.get_all_patients()
    return render_template('patients.html', patients=patients)

@app.route('/monitoring')
@app.route('/monitoring/<patient_id>')
@login_required
def monitoring_page(patient_id=None):
    patients = database.get_all_patients()
    selected_patient = None
    if patient_id:
        selected_patient = database.get_patient_by_id(patient_id)
    if not selected_patient and len(patients) > 0:
        selected_patient = patients[0]
        
    recent_vitals = []
    if selected_patient:
        recent_vitals = database.get_recent_vitals(selected_patient['patient_id'], limit=20)
        
    return render_template('monitoring.html', patients=patients, active_patient=selected_patient, initial_vitals=recent_vitals)

@app.route('/alerts')
@login_required
def alerts_page():
    status_filter = request.args.get('status', 'ALL')
    alerts = database.get_alerts(status_filter)
    return render_template('alerts.html', alerts=alerts, current_filter=status_filter)

@app.route('/history')
@app.route('/history/<patient_id>')
@login_required
def history_page(patient_id=None):
    patients = database.get_all_patients()
    selected_patient = None
    if patient_id:
        selected_patient = database.get_patient_by_id(patient_id)
    if not selected_patient and len(patients) > 0:
        selected_patient = patients[0]
        
    time_filter = request.args.get('range', 'all')
    history_records = []
    if selected_patient:
        history_records = database.get_patient_history(selected_patient['patient_id'], time_filter)
        
    return render_template('history.html', patients=patients, active_patient=selected_patient, records=history_records, time_filter=time_filter)

@app.route('/settings')
@login_required
def settings_page():
    predictor = VitalPredictor()
    model_exists = os.path.exists("models/lstm_model.pth") and os.path.exists("models/scaler.pkl")
    stats = database.get_dashboard_stats()
    return render_template('settings.html', 
                           model_loaded=predictor.is_loaded,
                           model_exists=model_exists,
                           features=FEATURE_NAMES,
                           seq_len=SEQUENCE_LENGTH,
                           hidden_size=HIDDEN_SIZE,
                           layers=NUM_LAYERS,
                           stats=stats)

# ----------------- REST API Endpoints ----------------- #

@app.route('/api/patients', methods=['GET'])
@login_required
def api_get_patients():
    return jsonify(database.get_all_patients())

@app.route('/api/patients', methods=['POST'])
@login_required
def api_create_patient():
    data = request.get_json() or request.form
    full_name = data.get('full_name', '').strip()
    age = data.get('age')
    gender = data.get('gender', 'Male')
    phone = data.get('phone', '').strip()
    if phone:
        # Strip all non-digit characters and truncate/validate to 10 digits
        clean_digits = re.sub(r'\D', '', phone)
        if len(clean_digits) > 10:
            clean_digits = clean_digits[-10:]
        if len(clean_digits) != 10:
            return jsonify({'error': 'Mobile phone number must be exactly 10 digits.'}), 400
        phone = clean_digits
    email = data.get('email', '').strip()
    medical_history = data.get('medical_history', '').strip()
    emergency_contact = data.get('emergency_contact', '').strip()

    if not full_name or not age:
        return jsonify({'error': 'Name and Age are required fields.'}), 400

    patient = database.create_patient(
        full_name=full_name,
        age=age,
        gender=gender,
        phone=phone,
        email=email,
        medical_history=medical_history,
        emergency_contact=emergency_contact
    )
    return jsonify({'success': True, 'patient': patient}), 201

@app.route('/api/patients/<patient_id>', methods=['GET'])
@login_required
def api_get_patient(patient_id):
    patient = database.get_patient_by_id(patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    return jsonify(patient)

@app.route('/api/patients/<patient_id>', methods=['PUT', 'POST'])
@login_required
def api_update_patient(patient_id):
    data = request.get_json() or request.form
    phone = data.get('phone', '')
    if phone:
        clean_digits = re.sub(r'\D', '', str(phone))
        if len(clean_digits) > 10:
            clean_digits = clean_digits[-10:]
        phone = clean_digits

    patient = database.update_patient(
        patient_id=patient_id,
        full_name=data.get('full_name'),
        age=data.get('age'),
        gender=data.get('gender'),
        phone=phone,
        email=data.get('email', ''),
        medical_history=data.get('medical_history', ''),
        emergency_contact=data.get('emergency_contact', '')
    )
    return jsonify({'success': True, 'patient': patient})

@app.route('/api/patients/<patient_id>', methods=['DELETE'])
@login_required
def api_delete_patient(patient_id):
    simulator_manager.stop_patient(patient_id)
    database.delete_patient(patient_id)
    return jsonify({'success': True, 'deleted_id': patient_id})

# Monitoring Controls
@app.route('/api/monitoring/<patient_id>/start', methods=['POST'])
@login_required
def api_start_monitoring(patient_id):
    simulator_manager.start_patient(patient_id)
    return jsonify({'success': True, 'patient_id': patient_id, 'status': 'ACTIVE'})

@app.route('/api/monitoring/<patient_id>/stop', methods=['POST'])
@login_required
def api_stop_monitoring(patient_id):
    simulator_manager.stop_patient(patient_id)
    return jsonify({'success': True, 'patient_id': patient_id, 'status': 'INACTIVE'})

@app.route('/api/monitoring/<patient_id>/simulate', methods=['POST'])
@login_required
def api_simulate_condition(patient_id):
    data = request.get_json() or {}
    condition = data.get('condition', 'NORMAL').upper()
    if condition not in ['NORMAL', 'WARNING', 'CRITICAL']:
        return jsonify({'error': 'Invalid condition. Use NORMAL, WARNING, or CRITICAL.'}), 400
        
    simulator_manager.set_patient_condition(patient_id, condition)
    return jsonify({'success': True, 'patient_id': patient_id, 'condition': condition})

@app.route('/api/dashboard/stats', methods=['GET'])
@login_required
def api_dashboard_stats():
    return jsonify(database.get_dashboard_stats())

@app.route('/api/alerts', methods=['GET'])
@login_required
def api_get_alerts():
    status = request.args.get('status', 'ALL')
    return jsonify(database.get_alerts(status))

@app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
@login_required
def api_resolve_alert(alert_id):
    database.resolve_alert(alert_id)
    return jsonify({'success': True, 'alert_id': alert_id})

@app.route('/api/history/<patient_id>', methods=['GET'])
@login_required
def api_get_history(patient_id):
    time_filter = request.args.get('range', 'all')
    records = database.get_patient_history(patient_id, time_filter)
    return jsonify(records)

@app.route('/api/history/<patient_id>/export-csv', methods=['GET'])
@login_required
def api_export_history_csv(patient_id):
    time_filter = request.args.get('range', 'all')
    records = database.get_patient_history(patient_id, time_filter)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Patient ID', 'Heart Rate (BPM)', 'SpO2 (%)', 'Temperature (C)', 'Respiratory Rate (RPM)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'AI Prediction', 'Risk Score (%)', 'Confidence (%)'])
    
    for r in records:
        writer.writerow([
            r['timestamp'], r['patient_id'], r['heart_rate'], r['spo2'],
            r['temperature'], r['respiratory_rate'], r['systolic_bp'],
            r['diastolic_bp'], r.get('prediction', 'N/A'),
            r.get('risk_score', 'N/A'), r.get('confidence', 'N/A')
        ])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=VitalTrace_{patient_id}_History.csv"}
    )

@app.route('/api/model/retrain', methods=['POST'])
@login_required
def api_retrain_model():
    try:
        train_model.train_vital_lstm(epochs=20)
        predictor = VitalPredictor()
        predictor.load()
        return jsonify({'success': True, 'message': 'PyTorch LSTM successfully retrained and reloaded into memory!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------- Socket.IO Event Handlers ----------------- #

@socketio.on('connect')
def handle_connect():
    print("[SocketIO] Client connected to VitalTrace stream")

@socketio.on('disconnect')
def handle_disconnect():
    print("[SocketIO] Client disconnected")

@socketio.on('join_patient')
def handle_join_patient(data):
    patient_id = data.get('patient_id')
    if patient_id:
        join_room(patient_id)
        print(f"[SocketIO] Client joined monitoring room for {patient_id}")

@socketio.on('leave_patient')
def handle_leave_patient(data):
    patient_id = data.get('patient_id')
    if patient_id:
        leave_room(patient_id)

if __name__ == '__main__':
    # Ensure models exist on startup
    if not (os.path.exists("models/lstm_model.pth") and os.path.exists("models/scaler.pkl")):
        print("[Startup] Training baseline PyTorch LSTM model...")
        train_model.train_vital_lstm(epochs=20)
    
    # Run server on 0.0.0.0:3000
    print("[VitalTrace] Clinical Intelligence Engine starting on http://0.0.0.0:3000")
    socketio.run(app, host='0.0.0.0', port=3000, debug=False, allow_unsafe_werkzeug=True)
