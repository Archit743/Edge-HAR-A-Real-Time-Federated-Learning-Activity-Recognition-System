from __future__ import annotations

import csv
from pathlib import Path

from flask import Flask, jsonify, render_template_string

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge-HAR Orchestrator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg: #000000;
            --surface: #0a0a0a;
            --border: #222222;
            --text: #ededed;
            --text-muted: #888888;
            --accent: #ffffff;
            --success: #10b981;
            --error: #ef4444;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', system-ui, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 48px;
            display: flex;
            flex-direction: column;
            gap: 32px;
            -webkit-font-smoothing: antialiased;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
        }

        h1 {
            font-weight: 500;
            font-size: 1.5rem;
            letter-spacing: -0.02em;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8125rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .dot {
            width: 6px;
            height: 6px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--success);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 24px;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 24px;
        }

        .card h3 {
            font-size: 0.8125rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .card .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 400;
            letter-spacing: -0.04em;
        }

        .card .delta {
            font-size: 0.75rem;
            margin-top: 8px;
            font-weight: 500;
        }

        .charts-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        .chart-container {
            height: 300px;
            position: relative;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8125rem;
        }

        th {
            text-align: left;
            padding: 12px 16px;
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
        }

        td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
        }

        tr:hover td {
            background: rgba(255,255,255,0.02);
        }

        .toggle-container {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8125rem;
            color: var(--text-muted);
            cursor: pointer;
        }

        input[type="checkbox"] {
            accent-color: var(--text);
            cursor: pointer;
        }
        
        .header-controls {
            display: flex;
            gap: 24px;
            align-items: center;
        }
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
                <div class="dot"></div> Server Active
            </div>
        </div>
    </header>

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
                    plugins: { 
                        legend: { display: false },
                        tooltip: { backgroundColor: '#000', titleColor: '#fff', bodyColor: color, borderColor: '#333', borderWidth: 1, cornerRadius: 4, displayColors: false }
                    },
                    scales: {
                        x: { grid: { color: '#111', drawBorder: false }, ticks: { maxRotation: 0 } },
                        y: { grid: { color: '#111', drawBorder: false } }
                    },
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

        let prevCount = 0;
        async function fetchMetrics() {
            try {
                const prevRes = await fetch('/api/metrics/previous');
                if (prevRes.ok) {
                    const pd = await prevRes.json();
                    accChart.data.datasets[1].data = pd.map(d => parseFloat(d.server_accuracy) || null);
                    lossChart.data.datasets[1].data = pd.map(d => parseFloat(d.server_loss) || null);
                }

                const res = await fetch('/api/metrics');
                const data = await res.json();
                if (!data.length || data.length === prevCount) return;
                prevCount = data.length;

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

            } catch(e) {}
        }

        fetchMetrics();
        setInterval(fetchMetrics, 2000);
    </script>
</body>
</html>
"""

def _read_rows(metrics_csv: Path) -> list[dict[str, str]]:
    if not metrics_csv.exists():
        return []
    with metrics_csv.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def create_dashboard_app(metrics_csv: Path) -> Flask:
    app = Flask(__name__)

    @app.get('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.get('/api/metrics')
    def metrics():
        return jsonify(_read_rows(metrics_csv))
        
    @app.get('/api/metrics/previous')
    def previous_metrics():
        prev_csv = metrics_csv.parent / f"{metrics_csv.stem}_previous{metrics_csv.suffix}"
        return jsonify(_read_rows(prev_csv))

    return app
