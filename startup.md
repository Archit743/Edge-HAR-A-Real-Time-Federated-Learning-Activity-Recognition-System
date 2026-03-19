# FedSense Demo Startup Guide

Follow these steps exactly to run the flawless showcase.

## 1. Clear Old Data
To start completely fresh on your phone, open Termux and run:
`rm phone/sensor_log.csv`

## 2. Start the Zero-Trust Hub (Laptop Terminal 1)
Run the central server that will mathematically manage the nodes:
`python -W ignore -m scripts.run_server`

## 3. Start the Live Dashboard (Laptop Terminal 2)
Boot up the sleek orchestrator UI:
`python -m scripts.run_dashboard`
*Open your browser to http://127.0.0.1:5000 and ensure "Show Baseline Reference" is set up.*

## 4. Connect the Simulated Node (Laptop Terminal 3)
Connect the massive baseline dataset representing your clean historical data:
`python -W ignore -m scripts.run_simulated_client --client-id 0`

## 5. Connect the Edge Phone (Termux)
While looking at the live dashboard on your laptop, hit enter on your phone to inject the edge anomaly:
`python -W ignore -m scripts.run_phone_client`

Watch the `FedAdam` math dynamically converge and block poisoned datasets!
