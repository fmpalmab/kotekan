import re
import pandas as pd
import matplotlib.pyplot as plt

log_path = "/home/juan_pablo/nvme.log"
disk_name = "nvme0n1"   # cámbialo si corresponde

rows = []
last_ts = None

# iostat -dx típicamente tiene una línea por disco con columnas:
# Device r/s rkB/s rrqm/s %rrqm r_await ra_size w/s wkB/s wrqm/s %wrqm w_await wa_size aqu-sz %util
# Pero puede variar según distro. Aquí extraemos timestamp + wkB/s o wMB/s si aparece.

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # timestamp al inicio (por ts)
        m_ts = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(.*)$", line)
        if m_ts:
            last_ts = m_ts.group(1)
            rest = m_ts.group(2)
        else:
            rest = line

        if last_ts is None:
            continue

        # línea del disco
        if not rest.startswith(disk_name + " "):
            continue

        parts = rest.split()
        # Heurística: buscar columna wkB/s o wMB/s según longitud
        # Intento común: wkB/s suele estar cerca de la mitad.
        # Tomamos el último %util para referencia y buscamos un campo que parezca "wkB/s".
        # Para robustez: probamos varias posiciones típicas.
        candidates_idx = [8, 9, 10]  # posiciones típicas donde cae wkB/s en muchas salidas
        wkbs = None
        for idx in candidates_idx:
            if idx < len(parts):
                try:
                    wkbs = float(parts[idx])
                    break
                except ValueError:
                    pass

        if wkbs is None:
            continue

        rows.append((last_ts, wkbs))

df = pd.DataFrame(rows, columns=["time", "wkB_s"])
df["time"] = pd.to_datetime(df["time"])
df["wMB_s"] = df["wkB_s"] / 1024.0
df["wGB_s"] = df["wMB_s"] / 1024.0

plt.figure()
plt.plot(df["time"], df["wGB_s"])
plt.xlabel("Time")
plt.ylabel("Write throughput (GB/s)")
plt.title(f"Disk write throughput over time ({disk_name})")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()
