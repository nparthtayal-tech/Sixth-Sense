import socket
import json
import time
import os

UDP_IP = "0.0.0.0"
UDP_PORT = 5000  # Default port used yesterday

print("Starting HyperIMU Bridge...")
print("Please ensure HyperIMU is sending UDP to your PC's IP.")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

for p in range(5000, 5020):
    try:
        sock.bind((UDP_IP, p))
        UDP_PORT = p
        break
    except OSError:
        continue

print(f"Bridge successfully listening on UDP {UDP_PORT}!")

last_print = time.time()
packet_count = 0

while True:
    try:
        data, addr = sock.recvfrom(1024)
        decoded = data.decode('utf-8', errors='ignore')
        
        ax, ay, az = 0.0, 0.0, 9.81
        
        # Try JSON
        try:
            payload = json.loads(decoded)
            if "ax" in payload:
                ax, ay, az = payload["ax"], payload["ay"], payload["az"]
            elif "accel" in payload:
                ax, ay, az = payload["accel"][0], payload["accel"][1], payload["accel"][2]
        except Exception:
            # Try CSV (HyperIMU)
            parts = decoded.replace('\r', '').replace('\n', '').split(',')
            vals = [float(p.strip()) for p in parts if p.strip()]
            if len(vals) >= 3:
                ax, ay, az = vals[-3], vals[-2], vals[-1]
                
        # Save to file for Streamlit to read safely
        with open("imu_data.json", "w") as f:
            json.dump({"ax": ax, "ay": ay, "az": az, "time": time.time()}, f)
            
        packet_count += 1
        if time.time() - last_print > 1.0:
            print(f"Receiving Data! ({packet_count} packets/sec) | Accel: {ax:.2f}, {ay:.2f}, {az:.2f}")
            packet_count = 0
            last_print = time.time()
            
    except Exception as e:
        print(f"Error parsing packet: {e}")
