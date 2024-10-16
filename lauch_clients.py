import subprocess
import time

# Server and client commands
server_command = ["python", "server.py"]
label_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
client_commands = [["python", "client.py", str(lp[0]), str(lp[1])] for lp in label_pairs]

# Start the server in a new console window
print("Starting the server...")
server_process = subprocess.Popen(server_command, creationflags=subprocess.CREATE_NEW_CONSOLE)

# Allow time for the server to start up
time.sleep(10)

# Start the clients in new console windows
client_processes = []
for i, client_command in enumerate(client_commands):
    print(f"Starting client {i} with labels {client_command[2]} and {client_command[3]}...")
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
