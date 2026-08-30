import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from model import VitalLSTM, FEATURE_NAMES, SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES

def generate_synthetic_vital_series(condition_type, length=SEQUENCE_LENGTH):
    """
    Generates a single continuous physiological time series of length `length`.
    Condition types:
      - 0: NORMAL (Healthy resting/stable vitals with subtle autonomic variation)
      - 1: WARNING (Pre-hypertension, mild tachycardia/bradycardia, mild hypoxia, low-grade fever)
      - 2: CRITICAL (Severe tachycardia/bradycardia, acute desaturation, respiratory distress, shock)
    """
    time_steps = np.arange(length)
    
    if condition_type == 0:  # NORMAL
        # Baselines
        base_hr = np.random.uniform(68, 82)
        base_spo2 = np.random.uniform(96.5, 99.0)
        base_temp = np.random.uniform(36.6, 37.1)
        base_rr = np.random.uniform(13, 17)
        base_sys = np.random.uniform(112, 122)
        base_dia = np.random.uniform(72, 79)
        
        # Smooth respiratory sinus arrhythmia & drift
        hr = base_hr + 2.5 * np.sin(time_steps * 0.8) + np.random.normal(0, 0.8, length)
        spo2 = base_spo2 + np.random.normal(0, 0.3, length)
        temp = base_temp + np.random.normal(0, 0.05, length)
        rr = base_rr + np.random.normal(0, 0.5, length)
        sys = base_sys + 2.0 * np.sin(time_steps * 0.4) + np.random.normal(0, 1.0, length)
        dia = base_dia + 1.2 * np.sin(time_steps * 0.4) + np.random.normal(0, 0.8, length)
        
    elif condition_type == 1:  # WARNING
        # Choose a clinical warning sub-pattern
        pattern = np.random.choice(['tachycardia_mild', 'hypoxia_mild', 'fever_mild', 'hypertension_mild', 'tachypnea_mild'])
        
        base_hr = np.random.uniform(70, 80)
        base_spo2 = np.random.uniform(96, 98)
        base_temp = np.random.uniform(36.8, 37.2)
        base_rr = np.random.uniform(14, 18)
        base_sys = np.random.uniform(115, 125)
        base_dia = np.random.uniform(75, 82)
        
        if pattern == 'tachycardia_mild':
            base_hr = np.random.uniform(105, 125)
        elif pattern == 'hypoxia_mild':
            base_spo2 = np.random.uniform(90.5, 94.0)
        elif pattern == 'fever_mild':
            base_temp = np.random.uniform(37.9, 38.8)
        elif pattern == 'hypertension_mild':
            base_sys = np.random.uniform(140, 158)
            base_dia = np.random.uniform(90, 98)
        elif pattern == 'tachypnea_mild':
            base_rr = np.random.uniform(22, 27)

        # Dynamic drift across sequence
        trend = np.linspace(0, np.random.uniform(1.0, 3.0), length)
        hr = base_hr + trend + np.random.normal(0, 1.5, length)
        spo2 = base_spo2 - 0.2 * trend + np.random.normal(0, 0.5, length)
        temp = base_temp + 0.05 * trend + np.random.normal(0, 0.08, length)
        rr = base_rr + 0.3 * trend + np.random.normal(0, 0.8, length)
        sys = base_sys + trend + np.random.normal(0, 1.8, length)
        dia = base_dia + 0.6 * trend + np.random.normal(0, 1.2, length)

    else:  # CRITICAL
        pattern = np.random.choice(['severe_hypoxia', 'severe_tachycardia', 'severe_bradycardia', 'severe_hypotension', 'hypertensive_crisis'])
        
        base_hr = np.random.uniform(75, 85)
        base_spo2 = np.random.uniform(96, 98)
        base_temp = np.random.uniform(36.8, 37.2)
        base_rr = np.random.uniform(15, 18)
        base_sys = np.random.uniform(120, 130)
        base_dia = np.random.uniform(75, 85)

        if pattern == 'severe_hypoxia':
            base_spo2 = np.random.uniform(78, 86)
            base_rr = np.random.uniform(28, 38)
            base_hr = np.random.uniform(115, 140)
        elif pattern == 'severe_tachycardia':
            base_hr = np.random.uniform(145, 180)
            base_sys = np.random.uniform(140, 170)
        elif pattern == 'severe_bradycardia':
            base_hr = np.random.uniform(34, 46)
            base_sys = np.random.uniform(75, 88)
            base_dia = np.random.uniform(45, 55)
        elif pattern == 'severe_hypotension':
            base_sys = np.random.uniform(65, 82)
            base_dia = np.random.uniform(40, 52)
            base_hr = np.random.uniform(120, 150)
            base_spo2 = np.random.uniform(85, 91)
        elif pattern == 'hypertensive_crisis':
            base_sys = np.random.uniform(180, 220)
            base_dia = np.random.uniform(115, 135)
            base_hr = np.random.uniform(105, 130)

        # Deterioration trend across the 10 timesteps
        deterioration = np.linspace(0, np.random.uniform(3.0, 8.0), length)
        hr = base_hr + deterioration + np.random.normal(0, 2.0, length)
        spo2 = np.clip(base_spo2 - 0.5 * deterioration + np.random.normal(0, 0.8, length), 60, 99)
        temp = base_temp + np.random.normal(0, 0.15, length)
        rr = base_rr + np.random.normal(0, 1.2, length)
        sys = base_sys + np.random.normal(0, 2.5, length)
        dia = base_dia + np.random.normal(0, 1.8, length)

    # Clip values to physically realistic boundaries
    hr = np.clip(hr, 30, 220)
    spo2 = np.clip(spo2, 60.0, 100.0)
    temp = np.clip(temp, 33.0, 42.0)
    rr = np.clip(rr, 6, 50)
    sys = np.clip(sys, 50, 240)
    dia = np.clip(dia, 30, 150)

    # Stack features: shape (length, 6)
    sequence = np.stack([hr, spo2, temp, rr, sys, dia], axis=1)
    return sequence

def create_dataset(samples_per_class=1200):
    """
    Generates a balanced dataset of multi-variate physiological time series.
    Total samples: 3 * samples_per_class (3,600 samples)
    """
    X = []
    y = []
    
    print(f"[Dataset] Generating {samples_per_class * 3} physiological time-series sequences...")
    for class_id in range(3):
        for _ in range(samples_per_class):
            seq = generate_synthetic_vital_series(class_id)
            X.append(seq)
            y.append(class_id)
            
    X = np.array(X, dtype=np.float32)  # Shape: (N, 10, 6)
    y = np.array(y, dtype=np.int64)    # Shape: (N,)
    return X, y

def train_vital_lstm(epochs=25, batch_size=32, lr=0.002):
    os.makedirs("models", exist_ok=True)
    
    print("==================================================")
    print("VitalTrace: Training PyTorch LSTM Health Predictor")
    print("==================================================")
    
    X, y = create_dataset(samples_per_class=1500)  # 4500 total sequences
    N, T, F = X.shape
    print(f"[Dataset] Features: {FEATURE_NAMES}")
    print(f"[Dataset] Shape: {N} samples, sequence length: {T}, features: {F}")
    
    # Flatten across time for StandardScaler fitting
    X_flat = X.reshape(-1, F)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    
    # Transform dataset
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, T, F).astype(np.float32)
    
    # Save Scaler
    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[Artifact] Saved StandardScaler to {scaler_path}")
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate Model, Loss & Optimizer
    model = VitalLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"[Architecture] 2-Layer LSTM (Hidden={HIDDEN_SIZE}) -> FC(32) -> FC(3)")
    print(f"[Training] Beginning {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        scheduler.step()
        train_acc = (correct / total) * 100.0
        train_loss = total_loss / total
        
        if epoch % 5 == 0 or epoch == epochs:
            # Evaluate on Test Set
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for val_x, val_y in test_loader:
                    v_out = model(val_x)
                    v_loss = criterion(v_out, val_y)
                    val_loss += v_loss.item() * val_x.size(0)
                    _, v_pred = torch.max(v_out, 1)
                    val_total += val_y.size(0)
                    val_correct += (v_pred == val_y).sum().item()
                    
            val_acc = (val_correct / val_total) * 100.0
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.2f}%")
            
    # Final Model Save
    model_path = "models/lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[Artifact] Saved trained PyTorch LSTM weights to {model_path}")
    print("[Success] PyTorch Model training completed successfully!")
    return True

if __name__ == '__main__':
    train_vital_lstm()
