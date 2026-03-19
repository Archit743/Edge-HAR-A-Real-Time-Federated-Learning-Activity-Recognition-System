import csv
import json
import subprocess
import time
from pathlib import Path

LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
]

def capture_sensor_data(label: str, duration: int):
    csv_path = Path("phone/sensor_log.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = csv_path.exists()
    
    print("\nInitializing Termux sensors... (Make sure Termux:API is installed from F-Droid!)")
    
    # We use a loop calling termux-sensor. 
    # To avoid process overhead on Android, fetching 10 samples at a time is efficient enough 
    # to yield a decent frequency without lagging the OS.
    
    start_time = time.time()
    
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "label", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        
        print(f"\nRecording started for {duration} seconds... Put the phone in your pocket!")
        
        try:
            while time.time() - start_time < duration:
                # Grab a batch of recent samples (10 updates)
                acc_proc = subprocess.run(["termux-sensor", "-s", "Accelerometer", "-n", "1"], capture_output=True, text=True)
                gyro_proc = subprocess.run(["termux-sensor", "-s", "Gyroscope", "-n", "1"], capture_output=True, text=True)
                
                if not acc_proc.stdout or not gyro_proc.stdout:
                    continue
                    
                acc_data = json.loads(acc_proc.stdout)
                gyro_data = json.loads(gyro_proc.stdout)
                
                # termux-sensor outputs {"Accelerometer": {"values": [...]}} 
                # but when returning multiple samples, it might wrap them in arrays or just give the latest
                # Just take the first valid one we get for synchronized writing:
                
                acc_vals = acc_data.get("Accelerometer", {}).get("values", [0.0, 0.0, 0.0])
                gyro_vals = gyro_data.get("Gyroscope", {}).get("values", [0.0, 0.0, 0.0])
                
                writer.writerow([
                    int(time.time() * 1000),
                    label,
                    acc_vals[0], acc_vals[1], acc_vals[2],
                    gyro_vals[0], gyro_vals[1], gyro_vals[2]
                ])
                f.flush()
                
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        finally:
            print(f"\nData appended to {csv_path}!")
            print("To stop continuous sensors if they get stuck, you can run 'termux-sensor -c'")

def main():
    print("=== FedSense Auto-Collector ===")
    print("Choose your activity label:")
    for i, label in enumerate(LABELS, 1):
        print(f"  {i}. {label}")
        
    try:
        choice = int(input("\nSelect number (1-6): ").strip())
        label = LABELS[choice - 1]
        duration = int(input("How many seconds to record? (e.g., 30): ").strip())
        
        print("\nGet ready... Recording begins in 3 seconds!")
        time.sleep(3)
        
        capture_sensor_data(label, duration)
        
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")

if __name__ == "__main__":
    main()
