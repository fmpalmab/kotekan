import requests
import numpy as np
import struct
FRAME_URL = "http://localhost:12048/inspect_frame/network_capture_buf"

PACKET_SIZE = 8256
HEADER_SIZE = 64 * 3  
SERIAL_OFFSET = 0   
SERIAL_FMT = ">Q"     

def get_frame():
    r = requests.get(FRAME_URL)
    r.raise_for_status()
    return np.frombuffer(r.content, dtype=np.uint8)

def extract_serial(packet):
    header = packet[:HEADER_SIZE]
    serial_bytes = header[SERIAL_OFFSET: SERIAL_OFFSET + struct.calcsize(SERIAL_FMT)]
    return struct.unpack(SERIAL_FMT, serial_bytes)[0]

def detect_packet_loss():
    frame = get_frame()
    n_packets = len(frame) // PACKET_SIZE
    serials = []
    for i in range(n_packets):
        pkt = frame[i*PACKET_SIZE : (i+1)*PACKET_SIZE]
        serials.append(extract_serial(pkt))    # detectar saltos
    lost = []
    for i in range(len(serials)-1):
        diff = serials[i+1] - serials[i]
        if diff != 1:
            lost.append((serials[i], serials[i+1], diff-1))   
        if not lost:
          print("No packet loss!")
    else:
        for s0, s1, n in lost:
            print(f"Loss: {n} packets between {s0} → {s1}")
        
        
detect_packet_loss()