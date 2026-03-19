# Edge-HAR Demonstration Run-Book

This is the official stage script and run-book for the federated learning presentation. It runs the full 3-node Edge-HAR architecture dynamically.

## Pre-Flight Check (Network)
**Critical Recommendation:** Turn on Windows "Mobile Hotspot" on your presentation laptop and connect your Android phone directly to it. This creates a secure, 0-latency tunnel and completely circumvents enterprise/university "Client Isolation" firewalls that would otherwise block port `8080`.

## 1. Laptop Server Startup
Open 3 independent terminal windows on your laptop. Wait for the `venv` to physically load on all of them.

**Terminal 1 (Zero-Trust Hub Orchestrator)**
```powershell
venv\Scripts\python.exe scripts\run_server.py
```

**Terminal 2 (Simulated Node 1)**
```powershell
venv\Scripts\python.exe scripts\run_simulated_client.py --client-id 0
```

**Terminal 3 (Simulated Node 2)**
```powershell
venv\Scripts\python.exe scripts\run_simulated_client.py --client-id 1
```

## 2. Dashboard Startup
Open a 4th terminal to start the WebSocket orchestrator backend.
```powershell
venv\Scripts\python.exe scripts\run_dashboard.py
```
* Open your browser to `http://127.0.0.1:5000`.
* The WebSockets will instantly connect.
* Observe the Dashboard Topology network map spinning up waiting for the Edge Node.

---

## 3. The Live Stage Play (Android Edge Node)
On your Android phone, open Termux and `cd FL`. Make sure you've run `git pull` so you have the latest scripts!

### Act I - The "Standard" Run (Valid Data)
Demonstrate an uninhibited Federated Learning aggregation.
1. Run the payload script to generate exactly 1,500 strings of valid mathematical walking sensor data:
   ```bash
   python scripts/generate_good_data.py
   ```
2. Start the edge node Python process:
   ```bash
   python scripts/run_phone_client.py
   ```
3. **Visuals:** Direct the judges to the Dashboard. As the phone aggregates its gradients perfectly alongside the two Simulated Nodes, the global `FedAdam` accuracy will steadily climb toward ~70% across 10 rounds as all 3 nodes contribute pristine mathematical structures.

### Act II - The "Byzantine Hack" Attack
Once Act I finishes, demonstrate the system's aggressive zero-trust security architecture.
1. On your phone, arm the corrupted payload that generates explosive tensors to mathematically simulate a malicious man-in-the-middle gradient injection:
   ```bash
   python scripts/generate_poison_data.py
   ```
2. **Restart the Server** on your laptop (Terminal 1) for a fresh run, and immediately launch the edge node on your phone again:
   ```bash
   python scripts/run_phone_client.py
   ```
3. **Visuals:** Watch the Dashboard Topology Map exactly as the server aggregates. The red malicious packets from the Edge Phone node will visibly hit an invisible mathematical Byzantine shield on the UI and shatter. The "Security Events" HUD metric will instantly iterate to `1` as the mathematical anomaly is effortlessly purged from the global `FedAdam` gradient update!
