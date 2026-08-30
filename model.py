import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# 6 Primary physiological features tracked in real-time
FEATURE_NAMES = [
    'heart_rate',       # BPM (normal: 60-100)
    'spo2',             # % (normal: 95-100)
    'temperature',      # Celsius (normal: 36.5-37.5)
    'respiratory_rate', # Breaths/min (normal: 12-20)
    'systolic_bp',      # mmHg (normal: 90-120)
    'diastolic_bp'      # mmHg (normal: 60-80)
]

SEQUENCE_LENGTH = 10
INPUT_SIZE = len(FEATURE_NAMES)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 3  # 0: NORMAL, 1: WARNING, 2: CRITICAL

CLASS_LABELS = {
    0: 'NORMAL',
    1: 'WARNING',
    2: 'CRITICAL'
}

class VitalLSTM(nn.Module):
    """
    PyTorch Deep Learning LSTM Architecture for Real-Time Physiological Time-Series Analysis.
    Input Shape: (Batch_Size, Sequence_Length=10, Features=6)
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super(VitalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 2-layer LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Fully connected classification head
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        last_timestep_out = out[:, -1, :]
        logits = self.fc_block(last_timestep_out)
        return logits


class VitalPredictor:
    """
    Inference engine that loads the trained PyTorch LSTM and Scaler,
    processes raw time-series vital sequences, and generates predictions.
    """
    _instance = None

    def __new__(cls, model_path="models/lstm_model.pth", scaler_path="models/scaler.pkl"):
        if cls._instance is None:
            cls._instance = super(VitalPredictor, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.scaler_path = scaler_path
            cls._instance.model = None
            cls._instance.scaler = None
            cls._instance.is_loaded = False
            cls._instance.load()
        return cls._instance

    def load(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.model = VitalLSTM()
                state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_loaded = True
                print(f"[PyTorch LSTM] Model and Scaler successfully loaded from {self.model_path}")
            else:
                print(f"[PyTorch LSTM] Model files not found ({self.model_path}, {self.scaler_path}). Run train_model.py first.")
                self.is_loaded = False
        except Exception as e:
            print(f"[PyTorch LSTM] Error loading model: {e}")
            self.is_loaded = False

    def predict_sequence(self, vitals_sequence):
        """
        Takes a list or array of recent vitals readings (at least 1 to 10 readings).
        Each reading is a dict or list with keys matching FEATURE_NAMES.
        Returns:
            dict: {
                'prediction': 'NORMAL' | 'WARNING' | 'CRITICAL',
                'risk_score': float (0.0 to 100.0),
                'confidence': float (0.0 to 100.0),
                'probabilities': {'NORMAL': float, 'WARNING': float, 'CRITICAL': float},
                'model_status': 'ACTIVE' | 'FALLBACK'
            }
        """
        if not self.is_loaded or self.model is None or self.scaler is None:
            self.load()
            if not self.is_loaded:
                # Return safe default if model not trained yet
                return {
                    'prediction': 'NORMAL',
                    'risk_score': 10.0,
                    'confidence': 50.0,
                    'probabilities': {'NORMAL': 0.7, 'WARNING': 0.2, 'CRITICAL': 0.1},
                    'model_status': 'TRAINING_REQUIRED'
                }

        try:
            # Format sequence into 2D array (N, 6)
            seq_data = []
            for item in vitals_sequence:
                if isinstance(item, dict):
                    row = [float(item.get(k, 0.0)) for k in FEATURE_NAMES]
                else:
                    row = [float(val) for val in item]
                seq_data.append(row)

            # Pad if fewer than SEQUENCE_LENGTH items (duplicate first item)
            while len(seq_data) < SEQUENCE_LENGTH:
                seq_data.insert(0, seq_data[0] if len(seq_data) > 0 else [75.0, 98.0, 36.8, 16.0, 115.0, 75.0])
            
            # Slice last SEQUENCE_LENGTH
            seq_data = seq_data[-SEQUENCE_LENGTH:]
            seq_np = np.array(seq_data, dtype=np.float32)

            # Preprocessing & Normalization using trained Scaler
            seq_scaled = self.scaler.transform(seq_np)  # (10, 6)

            # Convert to PyTorch Tensor: (1, 10, 6)
            tensor_input = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(tensor_input)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            p_normal = float(probs[0])
            p_warning = float(probs[1])
            p_critical = float(probs[2])

            pred_class_idx = int(np.argmax(probs))
            prediction_label = CLASS_LABELS[pred_class_idx]

            # Risk score calculation: weighted clinical risk formula
            # Normal contributes base risk, warning contributes mid, critical scales high
            raw_risk = (p_warning * 45.0) + (p_critical * 95.0) + (p_normal * 5.0)
            risk_score = round(min(100.0, max(0.0, raw_risk)), 1)
            
            # Confidence is highest softmax probability
            confidence = round(float(probs[pred_class_idx] * 100.0), 1)

            return {
                'prediction': prediction_label,
                'risk_score': risk_score,
                'confidence': confidence,
                'probabilities': {
                    'NORMAL': round(p_normal * 100.0, 1),
                    'WARNING': round(p_warning * 100.0, 1),
                    'CRITICAL': round(p_critical * 100.0, 1)
                },
                'model_status': 'ACTIVE'
            }
        except Exception as e:
            print(f"[PyTorch LSTM] Inference error: {e}")
            return {
                'prediction': 'NORMAL',
                'risk_score': 15.0,
                'confidence': 60.0,
                'probabilities': {'NORMAL': 70.0, 'WARNING': 20.0, 'CRITICAL': 10.0},
                'model_status': f'ERROR: {str(e)}'
            }
