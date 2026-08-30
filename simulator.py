import threading
import time
import random
import math
from datetime import datetime
import numpy as np

import database
from model import VitalPredictor, FEATURE_NAMES

class PatientVitalStream:
    """
    Simulates smooth, realistic physiological telemetry for a single patient
    and passes time-series windows to the PyTorch LSTM for real-time inference.
    """
    def __init__(self, patient_id, socketio_instance=None):
        self.patient_id = patient_id
        self.socketio = socketio_instance
        self.is_running = False
        self.thread = None
        self.condition = 'NORMAL'  # 'NORMAL' | 'WARNING' | 'CRITICAL'
        
        # Internal physiological state variables (Brownian / drift baseline)
        self.hr = random.uniform(70.0, 78.0)
        self.spo2 = random.uniform(97.5, 99.0)
        self.temp = random.uniform(36.7, 37.0)
        self.rr = random.uniform(14.0, 16.0)
        self.sys_bp = random.uniform(115.0, 122.0)
        self.dia_bp = random.uniform(74.0, 78.0)
        
        self.step_counter = 0
        self.recent_buffer = []  # Sliding window of 10 readings for LSTM
        self.last_alert_time = 0
        self.last_alert_type = None

    def set_condition(self, new_condition):
        """Changes simulated underlying physiological condition."""
        if new_condition in ['NORMAL', 'WARNING', 'CRITICAL']:
            self.condition = new_condition
            if new_condition == 'WARNING':
                self.hr = 115.0
                self.spo2 = 91.5
                self.temp = 38.3
                self.rr = 24.0
                self.sys_bp = 148.0
                self.dia_bp = 94.0
            elif new_condition == 'CRITICAL':
                self.hr = 158.0
                self.spo2 = 81.5
                self.temp = 39.8
                self.rr = 34.0
                self.sys_bp = 188.0
                self.dia_bp = 118.0
            elif new_condition == 'NORMAL':
                self.hr = 74.0
                self.spo2 = 98.2
                self.temp = 36.8
                self.rr = 15.0
                self.sys_bp = 118.0
                self.dia_bp = 76.0
            # Reset buffer to immediately reflect new physiological phase
            self.recent_buffer.clear()
            print(f"[Simulator] Patient {self.patient_id} simulated condition shifted to {new_condition}")

    def _generate_next_vitals(self):
        """Smooth autoregressive physiological update."""
        self.step_counter += 1
        t = self.step_counter
        
        # Sinusoidal autonomic components (e.g. respiratory sinus arrhythmia)
        hr_drift = 1.8 * math.sin(t * 0.4)
        bp_drift = 1.5 * math.sin(t * 0.2)

        if self.condition == 'NORMAL':
            target_hr = 74.0 + hr_drift
            target_spo2 = 98.0
            target_temp = 36.85
            target_rr = 15.0
            target_sys = 118.0 + bp_drift
            target_dia = 76.0 + 0.6 * bp_drift
            noise_scale = 0.5
            
        elif self.condition == 'WARNING':
            # Dynamic simulated moderate clinical stress / mild deterioration
            target_hr = 114.0 + hr_drift * 1.5
            target_spo2 = 92.5
            target_temp = 38.3
            target_rr = 24.0
            target_sys = 148.0 + bp_drift * 1.8
            target_dia = 94.0 + bp_drift
            noise_scale = 1.0
            
        else:  # CRITICAL
            # Simulated severe decompensation (acute distress / shock / tachy-arrhythmia)
            target_hr = 155.0 + hr_drift * 2.5
            target_spo2 = 82.0
            target_temp = 39.7
            target_rr = 34.0
            target_sys = 186.0 + bp_drift * 2.2
            target_dia = 118.0 + bp_drift * 1.5
            noise_scale = 1.8

        # Smooth gradual convergence to target physiological state (inertia)
        alpha = 0.25  # Convergence speed
        self.hr += alpha * (target_hr - self.hr) + random.gauss(0, 0.6 * noise_scale)
        self.spo2 += alpha * (target_spo2 - self.spo2) + random.gauss(0, 0.25 * noise_scale)
        self.temp += (alpha * 0.4) * (target_temp - self.temp) + random.gauss(0, 0.04 * noise_scale)
        self.rr += alpha * (target_rr - self.rr) + random.gauss(0, 0.4 * noise_scale)
        self.sys_bp += alpha * (target_sys - self.sys_bp) + random.gauss(0, 0.9 * noise_scale)
        self.dia_bp += alpha * (target_dia - self.dia_bp) + random.gauss(0, 0.6 * noise_scale)

        # Boundaries
        reading = {
            'heart_rate': round(max(30.0, min(220.0, self.hr)), 1),
            'spo2': round(max(60.0, min(100.0, self.spo2)), 1),
            'temperature': round(max(33.0, min(42.0, self.temp)), 1),
            'respiratory_rate': round(max(6.0, min(50.0, self.rr)), 1),
            'systolic_bp': round(max(50.0, min(240.0, self.sys_bp)), 1),
            'diastolic_bp': round(max(30.0, min(150.0, self.dia_bp)), 1),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        return reading

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        database.set_patient_monitoring_status(self.patient_id, 'ACTIVE')
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"[Simulator] Started real-time monitoring thread for Patient {self.patient_id}")

    def stop(self):
        self.is_running = False
        database.set_patient_monitoring_status(self.patient_id, 'INACTIVE')
        print(f"[Simulator] Stopped monitoring for Patient {self.patient_id}")

    def _run_loop(self):
        predictor = VitalPredictor()
        
        while self.is_running:
            try:
                # 1. Generate physiological vital reading
                vital = self._generate_next_vitals()
                
                # 2. Persist vital reading in SQLite
                database.record_vital_reading(
                    patient_id=self.patient_id,
                    hr=vital['heart_rate'],
                    spo2=vital['spo2'],
                    temp=vital['temperature'],
                    rr=vital['respiratory_rate'],
                    sys_bp=vital['systolic_bp'],
                    dia_bp=vital['diastolic_bp']
                )

                # 3. Buffer sequence for LSTM
                self.recent_buffer.append(vital)
                if len(self.recent_buffer) > 15:
                    self.recent_buffer.pop(0)

                # 4. PyTorch Deep Learning LSTM Inference
                prediction_result = predictor.predict_sequence(self.recent_buffer)

                # 5. Persist Prediction in SQLite
                pred_label = prediction_result['prediction']
                risk_score = prediction_result['risk_score']
                confidence = prediction_result['confidence']
                probs = prediction_result['probabilities']
                
                database.record_prediction(
                    patient_id=self.patient_id,
                    prediction=pred_label,
                    risk_score=risk_score,
                    confidence=confidence,
                    prob_normal=probs['NORMAL'],
                    prob_warning=probs['WARNING'],
                    prob_critical=probs['CRITICAL']
                )

                # 6. Generate Clinical Alert if Warning or Critical (with cooldown to prevent flooding)
                now = time.time()
                alert_payload = None
                if pred_label in ['WARNING', 'CRITICAL']:
                    if (now - self.last_alert_time > 12.0) or (self.last_alert_type != pred_label):
                        msg = self._format_alert_message(pred_label, vital, risk_score)
                        database.create_alert(
                            patient_id=self.patient_id,
                            alert_type=pred_label,
                            message=msg,
                            risk_score=risk_score,
                            confidence=confidence
                        )
                        self.last_alert_time = now
                        self.last_alert_type = pred_label
                        alert_payload = {
                            'patient_id': self.patient_id,
                            'alert_type': pred_label,
                            'message': msg,
                            'risk_score': risk_score,
                            'confidence': confidence,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                # 7. Broadcast real-time telemetry via Flask-SocketIO
                if self.socketio:
                    payload = {
                        'patient_id': self.patient_id,
                        'vital': vital,
                        'prediction': pred_label,
                        'risk_score': risk_score,
                        'confidence': confidence,
                        'probabilities': probs,
                        'condition': self.condition,
                        'alert': alert_payload
                    }
                    self.socketio.emit('vital_stream', payload)
                    if alert_payload:
                        self.socketio.emit('new_alert', alert_payload)

            except Exception as e:
                print(f"[Simulator] Error in monitoring loop for {self.patient_id}: {e}")

            time.sleep(1.5)  # 1.5 second continuous streaming interval

    def _format_alert_message(self, level, vital, risk):
        reasons = []
        if vital['heart_rate'] > 100 or vital['heart_rate'] < 55:
            reasons.append(f"HR {vital['heart_rate']} BPM")
        if vital['spo2'] < 95:
            reasons.append(f"SpO2 {vital['spo2']}%")
        if vital['temperature'] > 37.8:
            reasons.append(f"Temp {vital['temperature']}°C")
        if vital['respiratory_rate'] > 20 or vital['respiratory_rate'] < 10:
            reasons.append(f"RR {vital['respiratory_rate']} RPM")
        if vital['systolic_bp'] > 140 or vital['systolic_bp'] < 90:
            reasons.append(f"BP {vital['systolic_bp']}/{vital['diastolic_bp']} mmHg")
        
        detail = ", ".join(reasons) if reasons else "Multiple vital abnormalities detected"
        return f"{level} risk level detected ({detail}) — Risk Score: {risk}%"


class VitalSimulatorManager:
    """
    Singleton manager holding independent monitoring streams for all patients.
    """
    _instance = None

    def __new__(cls, socketio_instance=None):
        if cls._instance is None:
            cls._instance = super(VitalSimulatorManager, cls).__new__(cls)
            cls._instance.streams = {}
            cls._instance.socketio = socketio_instance
        elif socketio_instance and cls._instance.socketio is None:
            cls._instance.socketio = socketio_instance
        return cls._instance

    def set_socketio(self, socketio_instance):
        self.socketio = socketio_instance
        for stream in self.streams.values():
            stream.socketio = socketio_instance

    def start_patient(self, patient_id):
        if patient_id not in self.streams:
            self.streams[patient_id] = PatientVitalStream(patient_id, self.socketio)
        self.streams[patient_id].start()
        return True

    def stop_patient(self, patient_id):
        if patient_id in self.streams:
            self.streams[patient_id].stop()
        else:
            database.set_patient_monitoring_status(patient_id, 'INACTIVE')
        return True

    def set_patient_condition(self, patient_id, condition):
        if patient_id not in self.streams:
            self.streams[patient_id] = PatientVitalStream(patient_id, self.socketio)
            # If not already running, we can start it
            self.streams[patient_id].start()
        self.streams[patient_id].set_condition(condition)
        return True

    def get_status(self, patient_id):
        if patient_id in self.streams:
            stream = self.streams[patient_id]
            return {
                'patient_id': patient_id,
                'is_running': stream.is_running,
                'condition': stream.condition
            }
        return {'patient_id': patient_id, 'is_running': False, 'condition': 'NORMAL'}

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()
