import socket
import json
import time
import math
import random

UDP_IP = "127.0.0.1"
UDP_PORT = 5000

print(f"Starting UDP Sender to {UDP_IP}:{UDP_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

t = 0
while True:
    # Simulate some IMU noise and a slight wobble
    ax = random.gauss(0, 0.05)
    ay = math.sin(t) * 0.1 + random.gauss(0, 0.05)
    az = 9.81 + random.gauss(0, 0.05)
    
    packet = {
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0
    }
    
    data = json.dumps(packet).encode('utf-8')
    sock.sendto(data, (UDP_IP, UDP_PORT))
    
    t += 0.1
    time.sleep(0.1) # 10 Hz
