import subprocess
import time

server_command = ["python", "server.py"]
client_commands = [["python", "client.py", "--label", str(n)] for n in range(10)]

print("Starting the server...")
server_process = subprocess.Popen(server_command, creationflags=subprocess.CREATE_NEW_CONSOLE)

time.sleep(10)

client_processes = []
for i, client_command in enumerate(client_commands):
    print(f"Starting client {i} with label {client_command[3]}...")
    client_process = subprocess.Popen(client_command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    client_processes.append(client_process)
print("All clients started.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminating server and clients...")
    server_process.terminate()
    for client_process in client_processes:
        client_process.terminate()
    print("All processes terminated.")
