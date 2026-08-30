/**
 * VitalTrace - Real-Time Telemetry & PyTorch LSTM Inference Visualizer
 * Chart.js Multi-Stream Real-Time ICU Rendering Engine
 */

let charts = {};
const MAX_DATA_POINTS = 25;

// Initialize Chart.js configuration for smooth clinical dark charts
function initClinicalChart(canvasId, label, color, minVal, maxVal, unit) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: `${label} (${unit})`,
                data: [],
                borderColor: color,
                backgroundColor: color.replace(')', ', 0.08)').replace('rgb', 'rgba'),
                borderWidth: 2,
                pointRadius: 2.5,
                pointHoverRadius: 5,
                pointBackgroundColor: color,
                tension: 0.35,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 6 }
                },
                y: {
                    min: minVal,
                    max: maxVal,
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#fff',
                    bodyColor: color,
                    borderColor: '#334155',
                    borderWidth: 1
                }
            }
        }
    });

    return chart;
}

function initBloodPressureChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Systolic (mmHg)',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.08)',
                    borderWidth: 2,
                    pointRadius: 2.5,
                    tension: 0.35,
                    fill: false
                },
                {
                    label: 'Diastolic (mmHg)',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.08)',
                    borderWidth: 2,
                    pointRadius: 2.5,
                    tension: 0.35,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 6 }
                },
                y: {
                    min: 30,
                    max: 230,
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#fff',
                    borderColor: '#334155',
                    borderWidth: 1
                }
            }
        }
    });

    return chart;
}

// Push new real-time reading into chart
function appendChartReading(chart, timeLabel, value) {
    if (!chart) return;
    if (chart.data.labels.length >= MAX_DATA_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(value);
    chart.update('none');
}

function appendBPReading(chart, timeLabel, sys, dia) {
    if (!chart) return;
    if (chart.data.labels.length >= MAX_DATA_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }
    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(sys);
    chart.data.datasets[1].data.push(dia);
    chart.update('none');
}

// Initialize telemetry page
function initMonitoringDashboard(activePatientId, initialVitals = []) {
    // Initialize charts
    charts.hr = initClinicalChart('chart-hr', 'Heart Rate', 'rgb(239, 68, 68)', 40, 180, 'BPM');
    charts.spo2 = initClinicalChart('chart-spo2', 'SpO2 Oxygen', 'rgb(14, 165, 233)', 75, 100, '%');
    charts.temp = initClinicalChart('chart-temp', 'Core Temperature', 'rgb(245, 158, 11)', 34, 41, '°C');
    charts.rr = initClinicalChart('chart-rr', 'Respiratory Rate', 'rgb(16, 185, 129)', 8, 40, 'RPM');
    charts.bp = initBloodPressureChart('chart-bp');

    // Populate initial vitals if any exist in history
    if (initialVitals && initialVitals.length > 0) {
        initialVitals.forEach(v => {
            const timeStr = v.timestamp ? v.timestamp.split(' ')[1] || v.timestamp : '--:--:--';
            appendChartReading(charts.hr, timeStr, v.heart_rate);
            appendChartReading(charts.spo2, timeStr, v.spo2);
            appendChartReading(charts.temp, timeStr, v.temperature);
            appendChartReading(charts.rr, timeStr, v.respiratory_rate);
            appendBPReading(charts.bp, timeStr, v.systolic_bp, v.diastolic_bp);
        });
    }

    const activeSocket = window.socket || (typeof socket !== 'undefined' ? socket : null);
    if (activeSocket) {
        // Join room for active patient
        activeSocket.emit('join_patient', { patient_id: activePatientId });

        // Listen for live vitals on active patient
        activeSocket.on('vital_stream', (data) => {
            if (data.patient_id !== activePatientId) return;

            const v = data.vital;
            const timeLabel = v.timestamp ? (v.timestamp.split(' ')[1] || v.timestamp) : new Date().toLocaleTimeString();

            // 1. Update charts
            appendChartReading(charts.hr, timeLabel, v.heart_rate);
            appendChartReading(charts.spo2, timeLabel, v.spo2);
            appendChartReading(charts.temp, timeLabel, v.temperature);
            appendChartReading(charts.rr, timeLabel, v.respiratory_rate);
            appendBPReading(charts.bp, timeLabel, v.systolic_bp, v.diastolic_bp);

            // 2. Update live vital gauges
            const hrVal = document.getElementById('gauge-hr-val');
            const spo2Val = document.getElementById('gauge-spo2-val');
            const tempVal = document.getElementById('gauge-temp-val');
            const rrVal = document.getElementById('gauge-rr-val');
            const bpVal = document.getElementById('gauge-bp-val');

            if (hrVal) hrVal.innerText = v.heart_rate;
            if (spo2Val) spo2Val.innerText = `${v.spo2}%`;
            if (tempVal) tempVal.innerText = `${v.temperature}°C`;
            if (rrVal) rrVal.innerText = v.respiratory_rate;
            if (bpVal) bpVal.innerText = `${v.systolic_bp}/${v.diastolic_bp}`;

            // 3. Update AI Diagnostics Card (PyTorch LSTM inference)
            const predBadge = document.getElementById('ai-pred-badge');
            const riskVal = document.getElementById('ai-risk-score');
            const confVal = document.getElementById('ai-confidence-score');
            const riskBar = document.getElementById('ai-risk-bar');

            if (predBadge) {
                predBadge.className = `badge badge-${data.prediction.toLowerCase()}`;
                predBadge.innerText = data.prediction;
            }

            if (riskVal) riskVal.innerText = `${data.risk_score}%`;
            if (confVal) confVal.innerText = `${data.confidence}%`;

            if (riskBar) {
                riskBar.style.width = `${Math.min(100, Math.max(5, data.risk_score))}%`;
                riskBar.style.backgroundColor = data.prediction === 'CRITICAL' ? '#ef4444' :
                                               data.prediction === 'WARNING' ? '#f59e0b' : '#10b981';
            }

            // Update probability breakdown bars
            if (data.probabilities) {
                const pNorm = document.getElementById('prob-normal-bar');
                const pWarn = document.getElementById('prob-warning-bar');
                const pCrit = document.getElementById('prob-critical-bar');

                if (pNorm) {
                    pNorm.style.width = `${data.probabilities.NORMAL}%`;
                    pNorm.innerText = `${data.probabilities.NORMAL}%`;
                }
                if (pWarn) {
                    pWarn.style.width = `${data.probabilities.WARNING}%`;
                    pWarn.innerText = `${data.probabilities.WARNING}%`;
                }
                if (pCrit) {
                    pCrit.style.width = `${data.probabilities.CRITICAL}%`;
                    pCrit.innerText = `${data.probabilities.CRITICAL}%`;
                }
            }
        });
    }

    // Active polling fallback in case socket transport is delayed
    setInterval(async () => {
        try {
            const res = await fetch(`/api/patients/${activePatientId}/vitals?limit=1`);
            if (res.ok) {
                const data = await res.json();
                if (data.vitals && data.vitals.length > 0) {
                    const latest = data.vitals[0];
                    const timeLabel = latest.timestamp ? (latest.timestamp.split(' ')[1] || latest.timestamp) : new Date().toLocaleTimeString();
                    
                    const hrVal = document.getElementById('gauge-hr-val');
                    if (hrVal && (!hrVal.innerText || hrVal.innerText === '--')) {
                        appendChartReading(charts.hr, timeLabel, latest.heart_rate);
                        appendChartReading(charts.spo2, timeLabel, latest.spo2);
                        appendChartReading(charts.temp, timeLabel, latest.temperature);
                        appendChartReading(charts.rr, timeLabel, latest.respiratory_rate);
                        appendBPReading(charts.bp, timeLabel, latest.systolic_bp, latest.diastolic_bp);

                        hrVal.innerText = latest.heart_rate;
                        const spo2Val = document.getElementById('gauge-spo2-val');
                        if (spo2Val) spo2Val.innerText = `${latest.spo2}%`;
                        const tempVal = document.getElementById('gauge-temp-val');
                        if (tempVal) tempVal.innerText = `${latest.temperature}°C`;
                        const rrVal = document.getElementById('gauge-rr-val');
                        if (rrVal) rrVal.innerText = latest.respiratory_rate;
                        const bpVal = document.getElementById('gauge-bp-val');
                        if (bpVal) bpVal.innerText = `${latest.systolic_bp}/${latest.diastolic_bp}`;
                    }
                }
            }
        } catch(e) {}
    }, 2000);
}

// Trigger Simulated Condition (Normal, Warning, Critical)
async function triggerSimulation(patientId, condition) {
    try {
        const res = await fetch(`/api/monitoring/${patientId}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition })
        });
        const data = await res.json();
        if (data.success) {
            showToast(
                `Simulated condition set to ${condition}. Physiological telemetry adjusting...`,
                condition.toLowerCase() === 'normal' ? 'success' : condition.toLowerCase(),
                'SIMULATION CONTROLLER'
            );
        }
    } catch (err) {
        showToast(`Simulation trigger failed: ${err.message}`, 'critical');
    }
}
