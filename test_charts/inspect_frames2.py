import requests
import numpy as np
import matplotlib.pyplot as plt
import time# --- Parámetros ---
PAYLOAD_SIZE = 8256           # bytes por paquete
packets_per_frame = 512       # paquetes por frame
expected_bytes = PAYLOAD_SIZE * packets_per_frame

plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], lw=1)
ax.set_xlabel("Sample")
ax.set_ylabel("Power [dB]")
ax.grid(True, alpha=0.5)
ax.set_ylim(0, 25)
faxis = np.arange(8256)


while True:
    try:
        # Obtener frame desde Kotekan
        r = requests.get("http://localhost:12048/inspect_frame/network_capture_buf", timeout=2)
        raw_bytes = r.content
        print(f"Frame recibido: {len(raw_bytes)} bytes")        # Validar tamaño
        if len(raw_bytes) < expected_bytes:
            print(f"Frame incompleto ({len(raw_bytes)} < {expected_bytes}), descartando...")
            time.sleep(0.5)
            continue        
        
        # Dividir el frame en paquetes individuales
        pkts = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(packets_per_frame, PAYLOAD_SIZE)        # Procesar paquete por paquete
        for i, pkt in enumerate(pkts):
            real = ((pkt & 0xF0) >> 4).astype(np.int8)
            imag = (pkt & 0x0F).astype(np.int8)
            real[real >= 8] -= 16
            imag[imag >= 8] -= 16
            samples = real + 1j * imag
            power_dB = 10 * np.log10(np.abs(samples)**2 + 1e-6)
            line.set_data(np.arange(len(power_dB)), power_dB)
            ax.set_xlim(faxis[0], faxis[-1])
            ax.set_ylim(0, np.max(power_dB) + 5)
            fig.canvas.draw()
            fig.canvas.flush_events()      
            time.sleep(1)   
            
            print(f"Pkt {i+1}/{packets_per_frame} | Max power: {np.max(power_dB):.2f} dB")


    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)