import csv
import random
import time
from pathlib import Path

def main():
    csv_path = Path('phone/sensor_log.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating 1500 rows of CORRUPTED poison data to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'label', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        
        base_time = time.time()
        for i in range(1500):
            t = i * 0.02
            # Massive explosive tensors multiplied by 10,000 to trigger the Byzantine shield
            acc_x = random.uniform(-10000, 10000)
            acc_y = random.uniform(-10000, 10000)
            acc_z = random.uniform(-10000, 10000)
            gyro_x = random.uniform(-10000, 10000)
            gyro_y = random.uniform(-10000, 10000)
            gyro_z = random.uniform(-10000, 10000)
            
            # Switch between valid labels to radically confuse the network loss calculation
            label = str(random.choice([1, 2, 3, 4, 5, 6]))
            
            writer.writerow([base_time + t, label, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
            
    print("Done! Byzantine Poison payload armed on the Edge Node.")

if __name__ == '__main__':
    main()
