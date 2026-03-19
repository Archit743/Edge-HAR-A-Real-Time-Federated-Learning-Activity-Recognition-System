from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

from flask import Flask, render_template_string
from flask_socketio import SocketIO

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge-HAR Orchestrator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg: #000000; --surface: #0a0a0a; --border: #222222;
            --text: #ededed; --text-muted: #888888; --accent: #ffffff;
            --success: #10b981; --error: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background-color: var(--bg); color: var(--text);
            min-height: 100vh; padding: 48px; display: flex; flex-direction: column; gap: 32px;
            -webkit-font-smoothing: antialiased;
        }
        header { display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
        h1 { font-weight: 500; font-size: 1.5rem; letter-spacing: -0.02em; }
        .status-indicator { display: flex; align-items: center; gap: 8px; font-size: 0.8125rem; font-weight: 500; color: var(--text-muted); }
        .dot { width: 6px; height: 6px; background-color: var(--success); border-radius: 50%; box-shadow: 0 0 8px var(--success); }
        
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 24px; }
        .card h3 { font-size: 0.8125rem; font-weight: 500; color: var(--text-muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
        .card .value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 400; letter-spacing: -0.04em; }
        .card .delta { font-size: 0.75rem; margin-top: 8px; font-weight: 500; }
        
        .charts-section { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .chart-container { height: 300px; position: relative; }
        
        table { width: 100%; border-collapse: collapse; font-size: 0.8125rem; }
        th { text-align: left; padding: 12px 16px; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); }
        td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; }
        tr:hover td { background: rgba(255,255,255,0.02); }
        
        .toggle-container { display: flex; align-items: center; gap: 8px; font-size: 0.8125rem; color: var(--text-muted); cursor: pointer; }
        input[type="checkbox"] { accent-color: var(--text); cursor: pointer; }
        .header-controls { display: flex; gap: 24px; align-items: center; }
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Edge-HAR Orchestrator</h1>
            <div style="font-size: 0.875rem; color: var(--text-muted); margin-top: 4px;">Zero-Trust Hub ✦ FedAdam Active</div>
        </div>
        <div class="header-controls">
            <label class="toggle-container" title="Overlay dashed lines of the previous run for comparison">
                <input type="checkbox" id="togglePrev">
                Show Baseline Reference
            </label>
            <div class="status-indicator">
                <div class="dot"></div> <span id="socket-status">WS Connected</span>
            </div>
        </div>
    </header>
    
    <div class="card" style="padding: 24px; position: relative;">
        <h3 style="margin-bottom: 16px;">Live Network Topology</h3>
        <canvas id="networkMap" style="width: 100%; height: 160px; display: block;"></canvas>
    </div>

    <div class="metrics-grid">
        <div class="card">
            <h3>Iteration</h3>
            <div class="value" id="val-round">-</div>
            <div class="delta" id="val-nodes" style="color: var(--text-muted); font-family: 'Inter', sans-serif;">Waiting for nodes</div>
        </div>
        <div class="card">
            <h3>Global Accuracy</h3>
            <div class="value" id="val-acc" style="color: var(--accent);">-</div>
            <div class="delta" id="val-acc-delta">Validation Set Evaluation</div>
        </div>
        <div class="card">
            <h3>Global Loss</h3>
            <div class="value" id="val-loss">-</div>
            <div class="delta" id="val-loss-delta">Cross-Entropy</div>
        </div>
        <div class="card">
            <h3>Security Events</h3>
            <div class="value" id="val-security" style="color: var(--success);">0</div>
            <div class="delta" id="val-security-meta" style="color: var(--text-muted); font-family: 'Inter', sans-serif;">Anomalous nodes blocked</div>
        </div>
    </div>

    <div class="charts-section">
        <div class="card">
            <h3 style="margin-bottom: 16px;">Accuracy Convergence</h3>
            <div class="chart-container"><canvas id="accChart"></canvas></div>
        </div>
        <div class="card">
            <h3 style="margin-bottom: 16px;">Loss Descent</h3>
            <div class="chart-container"><canvas id="lossChart"></canvas></div>
        </div>
    </div>

    <div class="card" style="padding: 0; overflow-x: auto;">
        <table>
            <thead><tr id="table-headers"></tr></thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>

    <script>
        // --- Network Topology Canvas --- //
        const canvas = document.getElementById('networkMap');
        const ctx = canvas.getContext('2d');
        let cw = canvas.width = canvas.clientWidth;
        let ch = canvas.height = 160;

        window.addEventListener('resize', () => {
            cw = canvas.width = canvas.clientWidth;
            ch = canvas.height = 160;
        });

        const getNodes = () => [
            { id: 'hub', x: cw/2, y: ch/2, label: 'Central Aggregator', color: '#10b981', radius: 18 },
            { id: 'sim', x: cw*0.15, y: ch/2, label: 'Simulated Node', color: '#3b82f6', radius: 12 },
            { id: 'phone', x: cw*0.85, y: ch/2, label: 'Edge Phone', color: '#3b82f6', radius: 12 }
        ];

        let particles = [];
        
        function spawnParticles() {
            const nodes = getNodes();
            for(let i=0; i<10; i++) {
                particles.push({
                    x: nodes[1].x, y: nodes[1].y, tx: nodes[0].x, ty: nodes[0].y,
                    speed: 0.015 + Math.random()*0.02, progress: -Math.random()*0.5
                });
                particles.push({
                    x: nodes[2].x, y: nodes[2].y, tx: nodes[0].x, ty: nodes[0].y,
                    speed: 0.015 + Math.random()*0.02, progress: -Math.random()*0.5
                });
            }
        }

        function drawNetwork() {
            ctx.clearRect(0, 0, cw, ch);
            const nodes = getNodes();
            
            ctx.strokeStyle = '#222'; ctx.lineWidth = 1.5; ctx.beginPath();
            ctx.moveTo(nodes[1].x, nodes[1].y); ctx.lineTo(nodes[0].x, nodes[0].y);
            ctx.moveTo(nodes[2].x, nodes[2].y); ctx.lineTo(nodes[0].x, nodes[0].y);
            ctx.stroke();

            for (let i = particles.length - 1; i >= 0; i--) {
                let p = particles[i];
                p.progress += p.speed;
                if (p.progress > 1) { particles.splice(i, 1); continue; }
                if (p.progress > 0) {
                    let x = p.x + (p.tx - p.x) * p.progress;
                    let y = p.y + (p.ty - p.y) * p.progress;
                    ctx.fillStyle = '#fff';
                    ctx.shadowBlur = 8; ctx.shadowColor = '#3b82f6';
                    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI*2); ctx.fill();
                    ctx.shadowBlur = 0;
                }
            }

            nodes.forEach(n => {
                ctx.fillStyle = '#0a0a0a'; ctx.strokeStyle = n.color; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(n.x, n.y, n.radius, 0, Math.PI*2); ctx.fill(); ctx.stroke();
                ctx.fillStyle = '#888'; ctx.font = '11px JetBrains Mono'; ctx.textAlign = 'center';
                ctx.fillText(n.label, n.x, n.y + n.radius + 16);
            });
            requestAnimationFrame(drawNetwork);
        }
        drawNetwork();

        // --- Charts Setup --- //
        Chart.defaults.color = '#888';
        Chart.defaults.font.family = 'JetBrains Mono';
        Chart.defaults.font.size = 11;

        function createChart(ctx, color) {
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { data: [], borderColor: color, borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4, tension: 0.2 },
                        { data: [], borderColor: '#333', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0, tension: 0.2, hidden: true }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { backgroundColor: '#000', titleColor: '#fff', bodyColor: color, borderColor: '#333', borderWidth: 1, cornerRadius: 4, displayColors: false }},
                    scales: { x: { grid: { color: '#111', drawBorder: false }, ticks: { maxRotation: 0 } }, y: { grid: { color: '#111', drawBorder: false } }},
                    interaction: { mode: 'index', intersect: false }
                }
            });
        }

        const accChart = createChart(document.getElementById('accChart'), '#ffffff');
        const lossChart = createChart(document.getElementById('lossChart'), '#a1a1aa');

        document.getElementById('togglePrev').addEventListener('change', (e) => {
            accChart.data.datasets[1].hidden = !e.target.checked;
            lossChart.data.datasets[1].hidden = !e.target.checked;
            accChart.update(); lossChart.update();
        });

        // --- WebSocket Streaming --- //
        const socket = io();
        let prevCount = 0;

        socket.on('connect', () => {
            document.getElementById('socket-status').textContent = 'WS Connected';
            document.querySelector('.dot').style.backgroundColor = 'var(--success)';
            document.querySelector('.dot').style.boxShadow = '0 0 8px var(--success)';
            socket.emit('request_initial');
        });
        
        socket.on('disconnect', () => {
            document.getElementById('socket-status').textContent = 'WS Disconnected';
            document.querySelector('.dot').style.backgroundColor = 'var(--error)';
            document.querySelector('.dot').style.boxShadow = '0 0 8px var(--error)';
        });

        socket.on('metrics_update', (payload) => {
            const pd = payload.previous;
            accChart.data.datasets[1].data = pd.map(d => parseFloat(d.server_accuracy) || null);
            lossChart.data.datasets[1].data = pd.map(d => parseFloat(d.server_loss) || null);

            const data = payload.current;
            if (!data.length) return;
            
            // If new round mathematically aggregated, animate the topology!
            if (data.length > prevCount) {
                spawnParticles();
                prevCount = data.length;
            }

            const latest = data[data.length - 1];
            document.getElementById('val-round').textContent = latest.round || '-';

            let dAcc = 0, dLoss = 0;
            if (data.length > 1) {
                const prev = data[data.length - 2];
                if (latest.server_accuracy && prev.server_accuracy) dAcc = (parseFloat(latest.server_accuracy) - parseFloat(prev.server_accuracy)) * 100;
                if (latest.server_loss && prev.server_loss) dLoss = parseFloat(latest.server_loss) - parseFloat(prev.server_loss);
            }

            if (latest.server_accuracy) {
                document.getElementById('val-acc').textContent = (parseFloat(latest.server_accuracy) * 100).toFixed(2) + '%';
                const c = dAcc > 0 ? 'var(--success)' : (dAcc < 0 ? 'var(--error)' : 'var(--text-muted)');
                const sign = dAcc > 0 ? '+' : '';
                if (dAcc !== 0) document.getElementById('val-acc-delta').innerHTML = `<span style="color:${c}">${sign}${dAcc.toFixed(2)}%</span> step delta`;
            }

            if (latest.server_loss) {
                document.getElementById('val-loss').textContent = parseFloat(latest.server_loss).toFixed(4);
                const c = dLoss < 0 ? 'var(--success)' : (dLoss > 0 ? 'var(--error)' : 'var(--text-muted)');
                const sign = dLoss > 0 ? '+' : '';
                if (dLoss !== 0) document.getElementById('val-loss-delta').innerHTML = `<span style="color:${c}">${sign}${dLoss.toFixed(4)}</span> step delta`;
            }

            const s = data.reduce((a, r) => a + (parseInt(r.security_rejections) || 0), 0);
            document.getElementById('val-security').textContent = s;
            if (s > 0) {
                document.getElementById('val-security').style.color = 'var(--error)';
                document.getElementById('val-security-meta').innerHTML = '<span style="color:var(--error);font-weight:500;">Byzantine payload rejected</span>';
            }

            const pClients = parseInt(latest.participating_clients) || 0;
            const phones = parseInt(latest.phone_participated) || 0;
            const sims = pClients - phones;
            document.getElementById('val-nodes').innerHTML = `<span style="color:var(--text)">${sims} Sim, ${phones} Phone</span> Edge Targets`;

            accChart.data.labels = lossChart.data.labels = data.map(d => d.round);
            accChart.data.datasets[0].data = data.map(d => parseFloat(d.server_accuracy) || null);
            lossChart.data.datasets[0].data = data.map(d => parseFloat(d.server_loss) || null);
            accChart.update(); lossChart.update();

            const headers = Object.keys(data[0]);
            document.getElementById('table-headers').innerHTML = headers.map(h => `<th>${h.replace(/_/g, ' ')}</th>`).join('') + `<th>Δ Acc</th><th>Δ Loss</th>`;
            
            document.getElementById('table-body').innerHTML = [...data].reverse().map((r, i, arr) => {
                let cA = '<span style="color:var(--border)">-</span>', cL = '<span style="color:var(--border)">-</span>';
                if (i + 1 < arr.length) {
                    const pv = arr[i + 1];
                    if (r.server_accuracy && pv.server_accuracy) {
                        let d = (parseFloat(r.server_accuracy) - parseFloat(pv.server_accuracy)) * 100;
                        cA = `<span style="color:${d>0?'var(--success)':'var(--error)'}">${d>0?'+':''}${d.toFixed(2)}%</span>`;
                    }
                    if (r.server_loss && pv.server_loss) {
                        let d = parseFloat(r.server_loss) - parseFloat(pv.server_loss);
                        cL = `<span style="color:${d<0?'var(--success)':'var(--error)'}">${d>0?'+':''}${d.toFixed(4)}</span>`;
                    }
                }
                return `<tr>${headers.map(h => {
                    let content = (typeof r[h] === 'string' && r[h].includes('.') && !isNaN(r[h])) ? parseFloat(r[h]).toFixed(4) : r[h];
                    if (h === 'security_rejections' && parseInt(r[h]) > 0) {
                        content = `<span style="color:var(--error); font-weight:600;">🚫 ${r[h]} </span>`;
                    }
                    return `<td>${content}</td>`;
                }).join('')}<td>${cA}</td><td>${cL}</td></tr>`;
            }).join('');
        });
    </script>
</body>
</html>
"""

def _read_rows(metrics_csv: Path) -> list[dict[str, str]]:
    if not metrics_csv.exists():
        return []
    with metrics_csv.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def watch_metrics(app: Flask, socketio: SocketIO, metrics_csv: Path) -> None:
    last_mtime = 0.0
    while True:
        try:
            if metrics_csv.exists():
                mtime = metrics_csv.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    cur = _read_rows(metrics_csv)
                    prev_csv = metrics_csv.parent / f"{metrics_csv.stem}_previous{metrics_csv.suffix}"
                    prev = _read_rows(prev_csv)
                    socketio.emit('metrics_update', {'current': cur, 'previous': prev})
        except Exception:
            pass
        time.sleep(0.1)

def create_dashboard_app(metrics_csv: Path) -> tuple[Flask, SocketIO]:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'edge-har-secret'
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    @app.get('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @socketio.on('request_initial')
    def handle_initial():
        cur = _read_rows(metrics_csv)
        prev_csv = metrics_csv.parent / f"{metrics_csv.stem}_previous{metrics_csv.suffix}"
        prev = _read_rows(prev_csv)
        socketio.emit('metrics_update', {'current': cur, 'previous': prev})

    thread = threading.Thread(target=watch_metrics, args=(app, socketio, metrics_csv), daemon=True)
    thread.start()

    return app, socketio
