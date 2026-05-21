import time

def read_cpu():
    with open('/proc/stat', 'r', encoding='utf-8') as handle:
        fields = handle.readline().split()
    values = [int(value) for value in fields[1:8]]
    idle = values[3] + values[4]
    total = sum(values)
    return idle, total

idle1, total1 = read_cpu()
time.sleep(0.2)
idle2, total2 = read_cpu()
idle_delta = idle2 - idle1
total_delta = total2 - total1
usage = 0.0 if total_delta <= 0 else (1.0 - idle_delta / total_delta) * 100.0
print(f'CPU usage percent: {usage:.1f}')
