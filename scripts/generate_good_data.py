import csv
import math
import random
import time
from pathlib import Path

def main():
    csv_path = Path('phone/sensor_log.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating 1500 rows of GOOD synthetic data to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'label', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        
        base_time = time.time()
        for i in range(1500):
            # Realistic walking sine wave (1 Hz step frequency)
            t = i * 0.02
            acc_x = 1.0 + math.sin(t * 2 * math.pi) * 0.5 + random.uniform(-0.1, 0.1)
            acc_y = 9.8 + math.sin(t * 4 * math.pi) * 2.0 + random.uniform(-0.2, 0.2)
            acc_z = math.sin(t * 2 * math.pi) * 0.3 + random.uniform(-0.1, 0.1)
            
            gyro_x = math.cos(t * 2 * math.pi) * 0.5 + random.uniform(-0.05, 0.05)
            gyro_y = math.sin(t * 2 * math.pi) * 0.5 + random.uniform(-0.05, 0.05)
            gyro_z = random.uniform(-0.1, 0.1)
            
            writer.writerow([base_time + t, '1', acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
            
    print("Done! Valid data ready for training.")

if __name__ == '__main__':
    main()
