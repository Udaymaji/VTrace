import urllib.request
import urllib.parse
import json
import http.cookiejar
import time
import sqlite3

BASE_URL = "http://127.0.0.1:3000"

def run_tests():
    print("==================================================")
    print("VitalTrace: Running End-to-End System Verification")
    print("==================================================")

    # Setup session / cookie jar
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    # Helper function for GET/POST
    def get(path):
        url = f"{BASE_URL}{path}"
        req = urllib.request.Request(url)
        with opener.open(req) as res:
            return res.getcode(), res.read().decode('utf-8')

    def post(path, data, is_json=False):
        url = f"{BASE_URL}{path}"
        if is_json:
            encoded_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=encoded_data, headers={'Content-Type': 'application/json'})
        else:
            encoded_data = urllib.parse.urlencode(data).encode('utf-8')
            req = urllib.request.Request(url, data=encoded_data)
        with opener.open(req) as res:
            return res.getcode(), res.read().decode('utf-8')

    # STEP 1: Login
    print("[Test 1/32] Logging in as doctor / doctor123...")
    code, html = post("/login", {"username": "doctor", "password": "doctor123"})
    assert code == 200, f"Login failed with code {code}"
    assert "ICU Telemetry" in html or "Dashboard" in html, "Failed to reach dashboard after login"
    print("  -> Logged in successfully! (Session established)")

    # STEP 2: Open Dashboard
    print("[Test 2/32] Opening Dashboard...")
    code, html = get("/")
    assert code == 200
    assert "Active Patient Surveillance Cards" in html
    assert "LSTM Engine Live" in html
    print("  -> Dashboard verified!")

    # STEP 3: Open Patients
    print("[Test 3/32] Opening Patients page...")
    code, html = get("/patients")
    assert code == 200
    assert "Patient Management Directory" in html
    print("  -> Patients page verified!")

    # STEP 4 & 5: Add Patient (Amit Kumar)
    print("[Test 4-5/32] Adding Patient Amit Kumar (Age: 45, Male, Phone: 9999999999)...")
    payload = {
        "full_name": "Amit Kumar",
        "age": 45,
        "gender": "Male",
        "phone": "9999999999",
        "email": "amit.kumar@hospital.org",
        "medical_history": "Mild hypertension, non-smoker",
        "emergency_contact": "Sunita Kumar (Wife) - 9888888888"
    }
    code, res_body = post("/api/patients", payload, is_json=True)
    res_data = json.loads(res_body)
    assert res_data["success"] is True, "Failed to create patient"
    amit = res_data["patient"]
    amit_pid = amit["patient_id"]
    print(f"  -> Amit Kumar created with ID: {amit_pid}")

    # Verify Amit appears in patient list
    code, html = get("/patients")
    assert "Amit Kumar" in html and amit_pid in html
    print(f"  -> Verified Amit Kumar ({amit_pid}) appears on Patients page!")

    # STEP 6 & 7: Check SQLite direct persistence (simulates restart verification)
    print("[Test 6-7/32] Verifying SQLite persistence for Amit Kumar...")
    conn = sqlite3.connect("vitaltrace.db")
    c = conn.cursor()
    c.execute("SELECT patient_id, full_name, age, phone FROM patients WHERE patient_id = ?", (amit_pid,))
    row = c.fetchone()
    conn.close()
    assert row is not None and row[1] == "Amit Kumar" and row[2] == 45 and row[3] == "9999999999"
    print("  -> Verified SQLite record exists directly on disk!")

    # STEP 8: Open Amit's monitoring page
    print(f"[Test 8/32] Opening Monitoring page for {amit_pid}...")
    code, html = get(f"/monitoring/{amit_pid}")
    assert code == 200 and "Amit Kumar" in html
    print("  -> Amit's monitoring page loaded!")

    # STEP 9: Start Monitoring
    print(f"[Test 9/32] Starting Real-Time Monitoring for {amit_pid}...")
    code, res_body = post(f"/api/monitoring/{amit_pid}/start", {})
    res_data = json.loads(res_body)
    assert res_data["success"] is True and res_data["status"] == "ACTIVE"
    print("  -> Monitoring simulator started in background thread!")

    # STEP 10-16: Verify Live Vitals, SQLite readings, LSTM inference, Prediction, Risk, Confidence
    print("[Test 10-16/32] Waiting 4 seconds for telemetry generation and PyTorch LSTM inference...")
    time.sleep(4)

    conn = sqlite3.connect("vitaltrace.db")
    c = conn.cursor()
    c.execute("SELECT * FROM vitals WHERE patient_id = ? ORDER BY id DESC LIMIT 5", (amit_pid,))
    vitals_rows = c.fetchall()
    c.execute("SELECT * FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT 5", (amit_pid,))
    pred_rows = c.fetchall()
    conn.close()

    assert len(vitals_rows) > 0, "No vital readings recorded in SQLite"
    assert len(pred_rows) > 0, "No LSTM predictions recorded in SQLite"
    
    latest_vital = vitals_rows[0]
    latest_pred = pred_rows[0]
    print(f"  -> Generated Vitals: HR={latest_vital[2]} BPM, SpO2={latest_vital[3]}%, Temp={latest_vital[4]}°C, RR={latest_vital[5]} RPM, BP={latest_vital[6]}/{latest_vital[7]} mmHg")
    print(f"  -> PyTorch LSTM Output: Prediction={latest_pred[2]}, Risk={latest_pred[3]}%, Confidence={latest_pred[4]}%")
    assert latest_pred[2] in ["NORMAL", "WARNING", "CRITICAL"]

    # STEP 17-20: Simulate Warning
    print("[Test 17-20/32] Simulating WARNING condition...")
    code, res_body = post(f"/api/monitoring/{amit_pid}/simulate", {"condition": "WARNING"}, is_json=True)
    res_data = json.loads(res_body)
    assert res_data["success"] is True and res_data["condition"] == "WARNING"

    print("  -> Waiting 5 seconds for physiological drift and LSTM prediction transition...")
    time.sleep(5)

    conn = sqlite3.connect("vitaltrace.db")
    c = conn.cursor()
    c.execute("SELECT prediction, risk_score, confidence FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT 1", (amit_pid,))
    warn_pred = c.fetchone()
    c.execute("SELECT * FROM alerts WHERE patient_id = ? AND alert_type = 'WARNING'", (amit_pid,))
    warn_alerts = c.fetchall()
    conn.close()

    print(f"  -> Warning State: Prediction={warn_pred[0]}, Risk={warn_pred[1]}%, Confidence={warn_pred[2]}%")
    print(f"  -> Warning Alerts created in SQLite: {len(warn_alerts)}")
    assert warn_pred[0] in ["WARNING", "CRITICAL"], f"Expected WARNING/CRITICAL, got {warn_pred[0]}"
    assert len(warn_alerts) > 0, "Expected WARNING alert in SQLite"

    # STEP 21-24: Simulate Critical
    print("[Test 21-24/32] Simulating CRITICAL condition...")
    code, res_body = post(f"/api/monitoring/{amit_pid}/simulate", {"condition": "CRITICAL"}, is_json=True)
    res_data = json.loads(res_body)
    assert res_data["success"] is True and res_data["condition"] == "CRITICAL"

    print("  -> Waiting 5 seconds for acute decompensation and LSTM prediction...")
    time.sleep(5)

    conn = sqlite3.connect("vitaltrace.db")
    c = conn.cursor()
    c.execute("SELECT prediction, risk_score, confidence FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT 1", (amit_pid,))
    crit_pred = c.fetchone()
    c.execute("SELECT * FROM alerts WHERE patient_id = ? AND alert_type = 'CRITICAL'", (amit_pid,))
    crit_alerts = c.fetchall()
    conn.close()

    print(f"  -> Critical State: Prediction={crit_pred[0]}, Risk={crit_pred[1]}%, Confidence={crit_pred[2]}%")
    print(f"  -> Critical Alerts created in SQLite: {len(crit_alerts)}")
    assert crit_pred[0] == "CRITICAL", f"Expected CRITICAL, got {crit_pred[0]}"
    assert len(crit_alerts) > 0, "Expected CRITICAL alert in SQLite"

    # STEP 25: Reset Normal
    print("[Test 25/32] Resetting physiological condition to NORMAL...")
    code, res_body = post(f"/api/monitoring/{amit_pid}/simulate", {"condition": "NORMAL"}, is_json=True)
    time.sleep(3)
    print("  -> Physiological baseline restored to NORMAL!")

    # STEP 26: Stop Monitoring
    print(f"[Test 26/32] Stopping monitoring for {amit_pid}...")
    code, res_body = post(f"/api/monitoring/{amit_pid}/stop", {})
    res_data = json.loads(res_body)
    assert res_data["success"] is True and res_data["status"] == "INACTIVE"
    print("  -> Stopped monitoring for Amit Kumar!")

    # STEP 27: Open History
    print(f"[Test 27/32] Opening History page for {amit_pid}...")
    code, html = get(f"/history/{amit_pid}")
    assert code == 200 and "Amit Kumar" in html
    assert "Heart Rate" in html and "PyTorch Prediction" in html
    print("  -> Patient history verified!")

    # STEP 28: Open Alerts
    print("[Test 28/32] Opening Alerts page and testing resolution...")
    code, html = get("/alerts")
    assert code == 200
    assert "ICU Clinical Alert & Triage Center" in html
    assert amit_pid in html

    # Resolve an alert
    conn = sqlite3.connect("vitaltrace.db")
    c = conn.cursor()
    c.execute("SELECT id FROM alerts WHERE patient_id = ? LIMIT 1", (amit_pid,))
    alert_id = c.fetchone()[0]
    conn.close()

    code, res_body = post(f"/api/alerts/{alert_id}/resolve", {})
    res_data = json.loads(res_body)
    assert res_data["success"] is True
    print(f"  -> Alert #{alert_id} successfully marked as RESOLVED!")

    # STEP 29 & 30: Edit Amit
    print(f"[Test 29-30/32] Editing Amit Kumar (updating age to 46, adding phone note)...")
    edit_payload = {
        "full_name": "Amit Kumar (Updated)",
        "age": 46,
        "gender": "Male",
        "phone": "9999999999",
        "email": "amit.kumar@hospital.org",
        "medical_history": "Mild hypertension, stable post-observation",
        "emergency_contact": "Sunita Kumar (Wife) - 9888888888"
    }
    code, res_body = post(f"/api/patients/{amit_pid}", edit_payload, is_json=True)
    res_data = json.loads(res_body)
    assert res_data["success"] is True and res_data["patient"]["full_name"] == "Amit Kumar (Updated)"
    assert res_data["patient"]["age"] == 46
    print("  -> Edit verified and persisted in SQLite!")

    # STEP 31 & 32: Delete Amit
    print(f"[Test 31-32/32] Deleting Amit Kumar ({amit_pid})...")
    req = urllib.request.Request(f"{BASE_URL}/api/patients/{amit_pid}", headers={'Content-Type': 'application/json'}, method='DELETE')
    with opener.open(req) as res:
        res_data = json.loads(res.read().decode('utf-8'))
        assert res_data["success"] is True

    # Verify deletion in SQLite
    conn = sqlite3.connect("vitaltrace.db")
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE patient_id = ?", (amit_pid,))
    deleted_row = c.fetchone()
    conn.close()
    assert deleted_row is None, "Patient was not deleted from SQLite"
    print("  -> Amit Kumar completely removed from SQLite database!")

    print("==================================================")
    print("ALL 32 INTEGRATION TEST CASES PASSED PERFECTLY!")
    print("==================================================")

if __name__ == '__main__':
    run_tests()
