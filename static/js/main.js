/**
 * VitalTrace - Clinical Intelligence Frontend Engine
 * Real-Time Socket.IO Synchronization & Global UI Controllers
 */

// Initialize Socket.IO connection safely with fallback
let socket;
if (typeof io !== 'undefined') {
    try {
        socket = io({
            transports: ['websocket', 'polling'],
            reconnectionAttempts: 10,
            reconnectionDelay: 1000
        });
        window.socket = socket;
    } catch (e) {
        console.warn('[VitalTrace] Socket.IO initialization failed, using fallback:', e);
        socket = createFallbackSocket();
    }
} else {
    console.warn('[VitalTrace] Socket.IO library not loaded, using HTTP polling fallback');
    socket = createFallbackSocket();
}

function createFallbackSocket() {
    const listeners = {};
    const fallback = {
        on: (event, cb) => {
            if (!listeners[event]) listeners[event] = [];
            listeners[event].push(cb);
        },
        emit: (event, data) => {
            console.log('[FallbackSocket] emit:', event, data);
        },
        trigger: (event, data) => {
            if (listeners[event]) {
                listeners[event].forEach(cb => {
                    try { cb(data); } catch(err) { console.error(err); }
                });
            }
        }
    };
    window.socket = fallback;

    // HTTP polling fallback for live dashboard cards and alerts
    setInterval(async () => {
        try {
            const res = await fetch('/api/patients');
            if (res.ok) {
                const data = await res.json();
                if (data.patients) {
                    data.patients.forEach(p => {
                        fallback.trigger('vital_stream', {
                            patient_id: p.patient_id,
                            prediction: p.latest_prediction || 'NORMAL',
                            risk_score: p.latest_risk_score || 12.0,
                            confidence: p.latest_confidence || 98.0,
                            vital: {
                                heart_rate: p.latest_hr || 75,
                                spo2: p.latest_spo2 || 98,
                                temperature: p.latest_temp || 36.8,
                                respiratory_rate: p.latest_rr || 16,
                                systolic_bp: p.latest_sys || 120,
                                diastolic_bp: p.latest_dia || 80,
                                timestamp: new Date().toLocaleTimeString()
                            }
                        });
                    });
                }
            }
        } catch (e) {
            // silent catch on background poll
        }
    }, 2000);

    return fallback;
}

socket.on('connect', () => {
    console.log('[VitalTrace] WebSocket Connected to Clinical Telemetry Stream');
    const statusPill = document.getElementById('system-status-indicator');
    if (statusPill) {
        statusPill.innerHTML = '<span class="status-dot"></span> LSTM Engine Live';
    }
});

socket.on('disconnect', () => {
    console.warn('[VitalTrace] WebSocket Disconnected');
    const statusPill = document.getElementById('system-status-indicator');
    if (statusPill) {
        statusPill.innerHTML = '<span class="status-dot" style="background:#ef4444;"></span> Offline / Reconnecting';
    }
});

// Toast Notification Manager
function showToast(message, type = 'info', title = '') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = '🔔';
    if (type === 'critical') icon = '🚨';
    else if (type === 'warning') icon = '⚠️';
    else if (type === 'success') icon = '✅';

    toast.innerHTML = `
        <div style="font-size: 18px;">${icon}</div>
        <div style="flex:1;">
            ${title ? `<div style="font-weight:700; font-size:12px; text-transform:uppercase; margin-bottom:2px;">${title}</div>` : ''}
            <div>${message}</div>
        </div>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        setTimeout(() => toast.remove(), 400);
    }, 4500);
}

// Global Alert Listener from Socket.IO
socket.on('new_alert', (alert) => {
    const type = alert.alert_type.toLowerCase();
    showToast(
        `Patient ${alert.patient_id}: ${alert.message}`,
        type,
        `AI ALERT: ${alert.alert_type}`
    );

    // Update active badge in sidebar if exists
    const alertBadge = document.getElementById('sidebar-alert-badge');
    if (alertBadge) {
        let current = parseInt(alertBadge.innerText || '0', 10);
        alertBadge.innerText = current + 1;
        alertBadge.style.display = 'inline-block';
    }
});

// Real-Time Dashboard Card Updates
socket.on('vital_stream', (data) => {
    const pid = data.patient_id;
    
    // Update patient card if present on dashboard / patients page
    const card = document.getElementById(`patient-card-${pid}`);
    if (card) {
        // Update class status
        card.classList.remove('status-normal', 'status-warning', 'status-critical');
        card.classList.add(`status-${data.prediction.toLowerCase()}`);

        // Update badge
        const badge = card.querySelector('.status-badge');
        if (badge) {
            badge.className = `badge badge-${data.prediction.toLowerCase()} status-badge`;
            badge.innerText = data.prediction;
        }

        // Update vitals
        const v = data.vital;
        const hrEl = card.querySelector('.vital-hr');
        const spo2El = card.querySelector('.vital-spo2');
        const tempEl = card.querySelector('.vital-temp');
        const rrEl = card.querySelector('.vital-rr');
        const bpEl = card.querySelector('.vital-bp');

        if (hrEl) hrEl.innerText = `${v.heart_rate} BPM`;
        if (spo2El) spo2El.innerText = `${v.spo2}%`;
        if (tempEl) tempEl.innerText = `${v.temperature}°C`;
        if (rrEl) rrEl.innerText = `${v.respiratory_rate} RPM`;
        if (bpEl) bpEl.innerText = `${v.systolic_bp}/${v.diastolic_bp}`;

        // Update AI risk
        const riskEl = card.querySelector('.ai-risk-val');
        if (riskEl) {
            riskEl.innerText = `${data.risk_score}%`;
            riskEl.style.color = data.prediction === 'CRITICAL' ? 'var(--color-critical)' :
                                 data.prediction === 'WARNING' ? 'var(--color-warning)' : 'var(--color-normal)';
        }
    }
});

// Global Clock Update
function updateLiveClock() {
    const clockEl = document.getElementById('live-header-clock');
    if (clockEl) {
        const now = new Date();
        clockEl.innerText = now.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
        }) + ' • ' + now.toLocaleTimeString('en-US', { hour12: false });
    }
}
setInterval(updateLiveClock, 1000);
updateLiveClock();

// Modal Open / Close Helpers
function openModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('open');
}

function closeModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('open');
}

// Global API Helper for Start/Stop Monitoring
async function toggleMonitoring(patientId, startAction) {
    try {
        const url = `/api/monitoring/${patientId}/${startAction ? 'start' : 'stop'}`;
        const res = await fetch(url, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            showToast(
                `Patient ${patientId} monitoring ${startAction ? 'STARTED' : 'STOPPED'}`,
                'success'
            );
            setTimeout(() => window.location.reload(), 600);
        }
    } catch (err) {
        showToast(`Failed to toggle monitoring: ${err.message}`, 'critical');
    }
}
