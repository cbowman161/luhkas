import subprocess
result = subprocess.run(['cat', '/proc/meminfo'], capture_output=True, text=True)
print(result.stdout)