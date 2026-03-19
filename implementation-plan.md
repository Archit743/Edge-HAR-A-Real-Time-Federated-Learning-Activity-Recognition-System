# FedSense Implementation Plan

## Scope Lock
This implementation plan follows the agreed constraints:
1. UCI HAR label space only.
2. No synthetic data.
3. Manual phone labeling only.
4. Three clients total in demo scope (2 simulated + 1 phone).
5. Local WiFi accepted without transport encryption.

## Phase 1: Shared Contracts
### Objectives
1. Define a single 1D-CNN model architecture shared by all clients and server.
2. Implement weight serialization helpers (NumPy in, NumPy out).
3. Define typed runtime configuration schema.

### Tasks
1. Create model module with deterministic init and shape checks.
2. Add helper functions:
	- get_model_parameters(model) -> list[np.ndarray]
	- set_model_parameters(model, params) -> None
3. Define config keys:
	- rounds
	- local_epochs
	- learning_rate
	- strategy (fedavg or fedprox)
	- server_host and server_port
	- min_phone_windows_per_round
4. Add payload contract definition for client-server exchange.

### Exit Criteria
1. Model can round-trip parameters without shape mismatch.
2. Config loads from one source and is consumed by server and clients.

## Phase 2: Data Layer
### Simulated Clients Pipeline
1. Ingest UCI HAR.
2. Apply consistent preprocessing and normalization.
3. Partition into 2 non-IID splits via Dirichlet.
4. Save partition manifest for reproducibility.

### Phone Pipeline
1. Capture accelerometer and gyroscope in Termux logger.
2. Save local sensor records to CSV on phone storage.
3. Apply same windowing and normalization locally.
4. Keep all raw records local to phone.

### Exit Criteria
1. Simulated data loaders produce deterministic client splits.
2. Phone loader can build training windows from local CSV.

## Phase 3: Core Federated Runtime
### Objectives
1. Launch Flower server with FedAvg.
2. Run two simulated clients for 10 rounds.
3. Log global loss and accuracy after every round.

### Tasks
1. Implement server startup and round orchestration.
2. Implement simulated client fit and evaluate loops.
3. Persist per-round metrics to CSV.
4. Enforce payload contract on server input path.

### Exit Criteria
1. Simulated-only run completes 10 rounds.
2. Round metrics CSV is produced and complete.

## Phase 4: Real Phone Client
### Objectives
1. Connect Android Termux client to laptop server over local WiFi.
2. Train locally on phone windows.
3. Skip round gracefully when data is insufficient.

### Tasks
1. Implement manual session labeling workflow.
2. Wire phone client fit method to local dataset.
3. Return skip status with zero samples when below threshold.
4. Ensure skipped rounds do not block server round completion.

### Exit Criteria
1. Phone contributes to at least one round in a full run.
2. Skip path behaves correctly and preserves system stability.

## Phase 5: Privacy Boundary
### Objectives
1. Ensure only allowed payload fields cross network boundary.
2. Produce a post-run privacy report from server side evidence.

### Tasks
1. Add outbound validator on client side.
2. Add inbound validator on server side.
3. Add post-run scanner for:
	- raw sensor arrays
	- phone-local file paths
4. Generate one-page privacy artifact.

### Exit Criteria
1. Validators reject non-contract payloads.
2. Privacy report confirms no raw phone data server-side.

## Phase 6: Extensions (Config Switches)
### FedProx
1. Add strategy toggle between FedAvg and FedProx.
2. Store metrics in separate output files for comparison.

### Opacus DP-SGD (Phone Only)
1. Add optional DP training wrapper to phone training loop.
2. Record epsilon and DP settings as scalar metrics.

### Live Dashboard
1. Add local Flask page on laptop.
2. Show round number, loss, accuracy, phone participation, skipped rounds.

### Exit Criteria
1. Each extension can be enabled or disabled independently.
2. Extension outputs do not break baseline run.

## Phase 7: Demo Commands and Artifacts
### Objectives
1. Provide one command to start server + two simulated clients.
2. Document phone commands in README.
3. Produce all expected artifacts at run end.

### Required Artifacts
1. round_metrics.csv
2. convergence_fedavg_vs_fedprox.png
3. privacy_audit_report.md
4. centralized_baseline_metrics.csv

### Exit Criteria
1. Full workflow can be reproduced from clean workspace setup.
2. Artifacts are generated consistently.

## Verification Checklist
1. Simulated-only dry run: 10 rounds completed, CSV exists.
2. Full WiFi run: phone contributes at least once, skipped rounds tolerated.
3. Privacy check: no raw sensor data or phone paths in server outputs.
4. Strategy comparison: separate FedAvg and FedProx convergence outputs.
5. DP run: epsilon and utility delta recorded.
6. Dashboard: live updates over all 10 rounds.

## Risks and Mitigations
1. Phone does not collect enough windows in time.
	- Mitigation: configurable minimum window threshold and skip behavior.
2. Non-IID splits become too imbalanced.
	- Mitigation: persist manifest and tune Dirichlet concentration.
3. Contract drift between client and server payload format.
	- Mitigation: shared schema and explicit validators on both ends.

## Definition of Done
1. Core FL run (10 rounds) works with 2 simulated clients and 1 real phone client.
2. Privacy constraints are verifiably enforced and reported.
3. Optional extensions (FedProx, Opacus, dashboard) are switchable and functional.
4. Reproducible run instructions and artifacts are documented.

