# Edge-HAR

## Overview
Edge-HAR is a federated learning demo that trains a human activity recognition model across real devices on local WiFi. A laptop runs the aggregation server and two simulated clients. An Android phone running Termux acts as a real edge client.

The phone captures live accelerometer and gyroscope data, performs local training, and sends only model updates plus scalar metrics to the server. Raw sensor data never leaves the phone.

## Core Goals
1. Run a Flower-based federated training session for 10 rounds.
2. Use two simulated laptop clients built from UCI HAR data with non-IID Dirichlet splitting.
3. Include one real phone client that can participate or skip gracefully when local data is insufficient.
4. Log per-round global loss and accuracy.
5. Generate privacy evidence that server-side logs and artifacts contain no raw phone sensor data.

## System Topology
- Laptop:
	- Flower server (FedAvg baseline)
	- Two simulated clients (UCI HAR partitions)
	- Metrics logging, plotting, privacy checks
- Phone (Android + Termux):
	- Sensor logger for accelerometer and gyroscope
	- Local CSV buffer
	- Local preprocessing and training
	- Flower client communication over local WiFi

## Data Policy
- Simulated clients use only UCI HAR-derived data.
- Phone client uses only live device sensor data collected locally.
- No synthetic data generation.
- No auto-labeling. Labels are provided manually per recording session.

## Privacy Boundary
- Allowed payload fields from clients to server:
	- Model weight tensors
	- Sample counts
	- Scalar metrics
- Disallowed:
	- Raw time-series windows
	- Raw sensor buffers
	- Phone-local file paths
- Validation is enforced both outbound (client side) and inbound (server side).

## Demo Deliverables
After a full run, the project should produce:
1. Round metrics CSV.
2. Convergence chart.
3. Privacy audit report.
4. Centralized baseline result for comparison.

## Extensions (Optional, Switchable)
1. FedProx strategy as an alternative to FedAvg.
2. Opacus DP-SGD on phone client only, with epsilon reporting.
3. Local Flask dashboard for live training status.

## Out of Scope
1. Synthetic data or synthetic classes.
2. Auto-labeling or weak labeling on phone.
3. Transport encryption setup for this local demo.
4. Multi-machine simulation beyond the 2 simulated laptop clients + 1 phone client.

