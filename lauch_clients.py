import subprocess
import time

# Server and client commands
server_command = ["python", "server.py"]
client_commands = [["python", "client.py", str(n)] for n in range(10)]

# Start the server in a new console window
print("Starting the server...")
server_process = subprocess.Popen(server_command, creationflags=subprocess.CREATE_NEW_CONSOLE)

# Allow time for the server to start up
time.sleep(10)

# Start the clients in new console windows
client_processes = []
for i, client_command in enumerate(client_commands):
    print(f"Starting client {i} with label {client_command[2]}...")
    client_process = subprocess.Popen(client_command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    client_processes.append(client_process)

print("All clients started.")

# Keep the main script running to monitor processes (optional)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminating server and clients...")
    server_process.terminate()
    for client_process in client_processes:
        client_process.terminate()
    print("All processes terminated.")
